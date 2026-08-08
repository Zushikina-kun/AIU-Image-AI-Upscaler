"""
AI Image Upscaler  –  main.py
Tkinter GUI wrapper around realesrgan-ncnn-vulkan.exe

Features: thread-safe UI, settings persistence, auto model scan, drag-and-drop,
cancel support, dark/light theme, log panel, output dir picker, recursive folder
scan, per-format quality, tile/thread/GPU controls, pre/post processing,
background removal (rembg), subject detect+crop (YOLOv8), model descriptions,
configurable filename template, EXIF preservation.
"""
from __future__ import annotations

import json
import os
import platform
import queue
import re
import subprocess
import threading
import time
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
import tkinter.ttk as ttk

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    from PIL import Image, ImageTk, ImageFilter, ImageOps, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

# ultralytics imported lazily inside the worker to avoid slow startup
YOLO_AVAILABLE: bool | None = None   # None = not yet checked

import sys as _sys

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")
OUTPUT_FORMATS = ("png", "jpg", "webp")

# When frozen by PyInstaller (onedir), sys.executable is inside
# dist\AI Image Upscaler\ — which is exactly where we place models\ and
# realesrgan-ncnn-vulkan.exe.  When running from source, __file__ gives
# the project root.  Either way APP_DIR is the right base.
def _app_dir() -> Path:
    if getattr(_sys, "frozen", False):
        return Path(_sys.executable).parent
    return Path(__file__).parent

APP_DIR     = _app_dir()
CONFIG_FILE = APP_DIR / "settings.json"
MODELS_DIR  = APP_DIR / "models"
EXE_NAME    = "realesrgan-ncnn-vulkan.exe" if platform.system() == "Windows" \
              else "./realesrgan-ncnn-vulkan"

FIXED_SCALE_MODELS = {
    "realesr-animevideov3-x2": 2,
    "realesr-animevideov3-x3": 3,
    "realesr-animevideov3-x4": 4,
}

# ── Model descriptions ────────────────────────────────────────────────────────
MODEL_DESCRIPTIONS: dict[str, str] = {
    "realesrgan-x4plus": (
        "General-purpose 4× upscaler for real-world photos. "
        "Best all-rounder: handles JPEG compression, blur, and noise. "
        "23 RRDB blocks, ~64 MB. Use this when unsure."
    ),
    "realesrgan-x4plus-anime": (
        "Optimised for anime / line art / flat-colour illustrations. "
        "6 RRDB blocks — faster and smaller (~18 MB). "
        "Avoids over-sharpening smooth gradients that trip up the x4plus model."
    ),
    "realesrnet-x4plus": (
        "Lighter version of x4plus with less aggressive enhancement. "
        "Good when you want sharpening without hallucinating texture detail."
    ),
    "realesr-animevideov3-x2": (
        "Video-optimised anime model at 2× scale. "
        "Tuned for temporal consistency across frames — low flickering."
    ),
    "realesr-animevideov3-x3": (
        "Video-optimised anime model at 3× scale. "
        "Same temporal tuning as x2 variant, intermediate scale."
    ),
    "realesr-animevideov3-x4": (
        "Video-optimised anime model at 4× scale. "
        "The most commonly used animevideov3 variant for HD upscaling."
    ),
    "realesr-general-x4v3": (
        "Tiny/fast general model (~5 MB). "
        "Great for low-VRAM GPUs or when speed matters more than max quality."
    ),
    "realesr-general-wdn-x4v3": (
        "General-x4v3 with a built-in wavelet denoising component. "
        "Use when input images have heavy noise/grain."
    ),
    "4x-UltraSharp": (
        "Community favourite for AI-generated art and digital illustrations. "
        "Produces crisp, high-contrast edges. "
        "Can over-sharpen on natural photos — pair with a pre-denoise pass."
    ),
    "4x-UltraSharp-fp16": (
        "FP16 (half-precision) variant of 4x-UltraSharp. "
        "Smaller memory footprint on GPU, virtually identical quality. "
        "Preferred on cards with good fp16 throughput."
    ),
    "4x-UltraSharp-fp32": (
        "FP32 (full-precision) variant of 4x-UltraSharp. "
        "More numerically stable on older AMD GPUs (e.g. RX 580) "
        "where fp16 accumulation can produce soft/blurry output."
    ),
    "4x-UltraSharpV2": (
        "Updated V2 of UltraSharp — better texture fidelity and less ringing. "
        "Current recommended version for AI art and illustrations (2024+)."
    ),
    "4x-AnimeSharp": (
        "Alternative anime-optimised 4× model. "
        "Sharper edges than x4plus-anime with slightly more detail invention."
    ),
    "4x_NMKD-UltraYandere_300k": (
        "Community anime model trained on anime-style dataset. "
        "Strong detail enhancement for stylised art and manga."
    ),
    "4x_NMKD-YandereNeoXL_200k": (
        "Newer NMKD anime model with improved edge handling. "
        "Good alternative to UltraYandere for cleaner line art."
    ),
    "4x_BooruGan_650k": (
        "Trained heavily on Danbooru-style anime images. "
        "Excellent for anime fan art, soft shading, detailed hair."
    ),
    "2x-AnimeSharpV3": (
        "2× anime upscaler — use when you only need moderate scale-up "
        "without over-processing."
    ),
    "2x-sudo-RealESRGAN": (
        "2× general upscaler based on RealESRGAN architecture. "
        "Fast and clean for moderate upscaling."
    ),
    "2x_AniScale2_ESRGAN_i16_110K": (
        "2× anime model trained for 110K iterations. "
        "Smooth gradients with reasonable edge sharpness."
    ),
    "1x-UnResizeOnly_RCAN": (
        "1× scale — no upscaling. "
        "Uses RCAN to restore/denoise without changing resolution. "
        "Useful for cleaning up JPEG artefacts before a separate upscale pass."
    ),
    "RealESRGAN_x2plus": (
        "Official 2× Real-ESRGAN model for general photos. "
        "Equivalent quality to x4plus at half the scale factor."
    ),
    "RealESRGAN_x4plus": (
        "Official PyTorch weights for the x4plus model (same as the NCNN x4plus). "
        "Used by the model converter — not directly by the inference binary."
    ),
    "RealESRGAN_x4plus_anime_6B": (
        "Official PyTorch weights for the anime 6-block model. "
        "Used by the model converter — not directly by the inference binary."
    ),
}

DEFAULT_MODEL_DESC = (
    "Custom or community-trained model.\n"
    "Check the model filename or its source for usage guidance."
)

DEFAULT_SETTINGS: dict = {
    "model":          "realesrgan-x4plus",
    "scale":          4,
    "tile":           32,          # tile=32 measurably sharper on AMD Polaris/Vega
    "threads":        "2:2:2",
    "output_dir":     str(APP_DIR / "output"),
    "out_format":     "png",
    "jpeg_quality":   92,
    "webp_quality":   90,
    "phase2":         False,
    "model2":         "4x-UltraSharp-fp32",
    "dark_theme":     True,
    "gpu_id":         "0",
    "tta_mode":       False,
    # pre-processing
    "pre_denoise":    False,
    "denoise_h":      10,
    "pre_autolevels": False,
    "pre_jpeg_fix":   False,
    # post-processing
    "post_sharpen":   True,         # ON by default — corrects ESRGAN's intentional softness
    "sharpen_preset": "Photo",      # Photo / Anime / Art / Strong / Custom
    "sharpen_radius": 1.5,
    "sharpen_pct":    130,          # slightly stronger default than before
    "sharpen_thresh": 3,
    "post_contrast":  False,
    "contrast_factor":1.05,
    # tools
    "bg_remove":      False,
    "smart_crop":     False,
    "smart_pad":      True,
    "detect_classes": "person,animal",
    "preserve_exif":  True,
    "name_template":  "{stem}_{model}_{scale}x",
}

# ── Sharpen presets ───────────────────────────────────────────────────────────
# (radius, percent, threshold)  — tuned from test data on real ESRGAN output
SHARPEN_PRESETS: dict[str, tuple[float, int, int]] = {
    "Photo":  (1.5, 130, 3),   # natural photos — moderate crisp, no ringing
    "Anime":  (1.0, 150, 2),   # anime/line art — sharper edges, low threshold
    "Art":    (2.0, 160, 2),   # AI-generated art / illustrations — strong detail
    "Strong": (2.0, 200, 1),   # maximum sharpening — use on very soft outputs
    "Custom": None,             # user-defined values from spinboxes
}

# ── Theme palettes ────────────────────────────────────────────────────────────
DARK = {
    "bg": "#1e1e2e", "fg": "#cdd6f4", "panel": "#313244",
    "accent": "#89b4fa", "success": "#a6e3a1", "warn": "#f9e2af",
    "error": "#f38ba8", "entry": "#45475a", "select": "#585b70",
    "border": "#585b70",
}
LIGHT = {
    "bg": "#f5f5f5", "fg": "#333333", "panel": "#e0e0e0",
    "accent": "#1976d2", "success": "#2e7d32", "warn": "#e65100",
    "error": "#c62828", "entry": "#ffffff", "select": "#bbdefb",
    "border": "#bdbdbd",
}


def apply_theme(root: tk.Tk, dark: bool) -> dict:
    c = DARK if dark else LIGHT
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=c["bg"], foreground=c["fg"],
                 fieldbackground=c["entry"], troughcolor=c["panel"],
                 selectbackground=c["select"], selectforeground=c["fg"],
                 bordercolor=c["border"], relief="flat")
    for w in ("TFrame", "TLabelframe"):
        s.configure(w, background=c["bg"])
    s.configure("TLabelframe.Label", background=c["bg"], foreground=c["accent"])
    s.configure("TLabel",      background=c["bg"], foreground=c["fg"])
    s.configure("TButton",     background=c["panel"], foreground=c["fg"], padding=(6, 3))
    s.map("TButton",
          background=[("active", c["accent"]), ("pressed", c["select"])],
          foreground=[("active", "#ffffff")])
    s.configure("Accent.TButton", background=c["accent"], foreground="#ffffff",
                 font=("Segoe UI", 9, "bold"))
    s.map("Accent.TButton", background=[("active", c["select"])])
    s.configure("TCombobox", fieldbackground=c["entry"], background=c["entry"],
                 foreground=c["fg"], arrowcolor=c["fg"])
    s.configure("TCheckbutton", background=c["bg"], foreground=c["fg"])
    s.configure("TScale", background=c["bg"], troughcolor=c["entry"])
    s.configure("Horizontal.TProgressbar", troughcolor=c["panel"], background=c["accent"])
    s.configure("TNotebook", background=c["bg"], tabmargins=[0, 0, 0, 0])
    s.configure("TNotebook.Tab", background=c["panel"], foreground=c["fg"], padding=(10, 4))
    s.map("TNotebook.Tab",
          background=[("selected", c["accent"])], foreground=[("selected", "#ffffff")])
    root.configure(bg=c["bg"])
    return c


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_image(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_EXTS)


def scan_models() -> list[str]:
    names: list[str] = []
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.glob("*.param")):
            names.append(p.stem)
    for b in ("realesrgan-x4plus", "realesrgan-x4plus-anime", "realesr-animevideov3-x4"):
        if b not in names:
            names.insert(0, b)
    return names


def load_settings() -> dict:
    merged = dict(DEFAULT_SETTINGS)
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            merged.update(saved)
        except Exception:
            pass

    # ── Sanity-check output_dir ───────────────────────────────────────────────
    # If the saved output_dir is inside a path that no longer exists (e.g. an
    # old project root from a dev build that's now a frozen EXE), reset it to
    # the default so the user doesn't get silent write failures.
    out = Path(merged["output_dir"])
    try:
        # Accept if the dir exists OR if its parent exists (can be created).
        # Reject only if neither half makes sense on this system.
        if not out.exists() and not out.parent.exists():
            merged["output_dir"] = DEFAULT_SETTINGS["output_dir"]
    except Exception:
        merged["output_dir"] = DEFAULT_SETTINGS["output_dir"]

    return merged


def save_settings(cfg: dict) -> None:
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


def human_size(path: str) -> str:
    try:
        b = os.path.getsize(path)
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"
    except OSError:
        return "?"


def image_info(path: str) -> str:
    try:
        if not PIL_AVAILABLE:
            return human_size(path)
        with Image.open(path) as im:
            w, h = im.size
        return f"{w}×{h}  {human_size(path)}"
    except Exception:
        return human_size(path)


def format_name(template: str, stem: str, model: str, scale: int, ext: str) -> str:
    """Apply the filename template. Sanitise for filesystem safety."""
    name = template.format(stem=stem, model=model, scale=scale, ext=ext)
    # Remove characters invalid on Windows
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name


def copy_exif(src: str, dst: str) -> None:
    """Copy EXIF data from src to dst (JPEG only; silently no-op otherwise)."""
    if not PIEXIF_AVAILABLE:
        return
    try:
        exif_bytes = piexif.load(src)
        out_bytes = piexif.dump(exif_bytes)
        piexif.insert(out_bytes, dst)
    except Exception:
        pass


# ── Image processing helpers (pre / post / tools) ─────────────────────────────
def preprocess_image(src: str, cfg: dict, log_fn) -> str:
    """
    Apply enabled pre-processing steps to src, save to a temp file,
    return the temp file path (or src if nothing was done).
    """
    if not PIL_AVAILABLE:
        return src
    did_something = False
    img: Image.Image | None = None

    def _open():
        nonlocal img
        if img is None:
            img = Image.open(src).convert("RGB")

    # 1 ── Auto-levels (contrast normalisation)
    if cfg.get("pre_autolevels"):
        _open()
        img = ImageOps.autocontrast(img, cutoff=0.5)
        did_something = True
        log_fn("    Pre: auto-levels applied", "info")

    # 2 ── OpenCV Non-Local Means denoising
    if cfg.get("pre_denoise") and CV2_AVAILABLE:
        _open()
        import cv2
        import numpy as np
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h_val = int(cfg.get("denoise_h", 10))
        arr = cv2.fastNlMeansDenoisingColored(arr, None, h_val, h_val, 7, 21)
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        did_something = True
        log_fn(f"    Pre: NLM denoise h={h_val}", "info")
    elif cfg.get("pre_denoise") and not CV2_AVAILABLE:
        log_fn("    Pre: denoise skipped (opencv not installed)", "warn")

    # 3 ── JPEG artifact softening (mild median-like blur before upscale)
    if cfg.get("pre_jpeg_fix"):
        _open()
        img = img.filter(ImageFilter.MedianFilter(size=3))
        did_something = True
        log_fn("    Pre: JPEG artifact reduction applied", "info")

    if not did_something or img is None:
        return src

    # Save to a temp file in the output dir
    out_dir = Path(cfg.get("output_dir", str(APP_DIR / "output")))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    tmp = str(out_dir / f"_pre_{stem}.png")
    img.save(tmp, "PNG")
    img.close()
    return tmp


def postprocess_image(src: str, cfg: dict, log_fn) -> str:
    """
    Apply enabled post-processing steps to src in-place (overwrites src).
    Returns src path.
    """
    if not PIL_AVAILABLE:
        return src
    did_something = False

    try:
        img = Image.open(src).convert("RGB")
    except Exception as e:
        log_fn(f"    Post: could not open {src}: {e}", "warn")
        return src

    # 1 ── Unsharp mask sharpening
    if cfg.get("post_sharpen"):
        radius = float(cfg.get("sharpen_radius", 1.5))
        pct    = int(cfg.get("sharpen_pct", 120))
        thresh = int(cfg.get("sharpen_thresh", 3))
        img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=pct, threshold=thresh))
        did_something = True
        log_fn(f"    Post: sharpen r={radius} pct={pct}%", "info")

    # 2 ── Contrast enhancement
    if cfg.get("post_contrast"):
        factor = float(cfg.get("contrast_factor", 1.05))
        img = ImageEnhance.Contrast(img).enhance(factor)
        did_something = True
        log_fn(f"    Post: contrast ×{factor}", "info")

    if did_something:
        img.save(src, "PNG")
    img.close()
    return src


def remove_background(src: str, out_dir: str, stem: str, log_fn) -> str | None:
    """Remove background using rembg. Returns output path or None on failure."""
    if not REMBG_AVAILABLE:
        log_fn("    BG remove: rembg not installed. Run: pip install 'rembg[cpu]'", "warn")
        return None
    try:
        log_fn("    BG remove: running (first run downloads ~170MB model)…", "info")
        with open(src, "rb") as f:
            data = f.read()
        result = rembg.remove(data)
        out_path = os.path.join(out_dir, f"{stem}_nobg.png")
        with open(out_path, "wb") as f:
            f.write(result)
        log_fn(f"    BG remove: saved {os.path.basename(out_path)}", "success")
        return out_path
    except Exception as e:
        log_fn(f"    BG remove failed: {e}", "error")
        return None


def detect_and_crop(src: str, cfg: dict, log_fn) -> str | None:
    """
    Detect subject (person/animal/object) via YOLOv8n, crop to the largest
    detected bounding box with padding, return cropped path or None.
    """
    global YOLO_AVAILABLE
    if YOLO_AVAILABLE is None:
        try:
            from ultralytics import YOLO as _YOLO  # noqa: F401
            YOLO_AVAILABLE = True
        except ImportError:
            YOLO_AVAILABLE = False

    if not YOLO_AVAILABLE:
        log_fn("    Smart crop: ultralytics not installed. Run: pip install ultralytics", "warn")
        return None

    if not PIL_AVAILABLE:
        return None

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")   # auto-downloads ~6 MB on first use

        # Parse which class names to prioritise
        want = [c.strip().lower() for c in cfg.get("detect_classes", "person,animal").split(",")]

        results = model(src, verbose=False)
        if not results or not results[0].boxes:
            log_fn("    Smart crop: no objects detected", "warn")
            return None

        boxes   = results[0].boxes
        names   = results[0].names   # {id: name}

        # Score boxes: prioritise wanted classes, then by confidence
        best = None
        best_score = -1.0
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, "").lower()
            conf = float(box.conf[0])
            # Check if class is a wanted category (partial match: "animal" matches "cat","dog",etc.)
            wanted = any(
                w in cls_name or cls_name in w or
                (w == "animal" and cls_name in {
                    "cat","dog","bird","horse","sheep","cow","elephant","bear",
                    "zebra","giraffe","rabbit","mouse","hamster","parrot"})
                for w in want
            )
            score = conf * (2.0 if wanted else 1.0)
            if score > best_score:
                best_score = score
                best = box

        if best is None:
            log_fn("    Smart crop: no matching object found", "warn")
            return None

        x1, y1, x2, y2 = (int(v) for v in best.xyxy[0])
        cls_name = names.get(int(best.cls[0]), "object")
        log_fn(f"    Smart crop: detected '{cls_name}' conf={float(best.conf[0]):.2f}", "info")

        with Image.open(src) as im:
            w, h = im.size
            pad = int(min(w, h) * 0.10)   # 10% padding
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            if cfg.get("smart_pad"):
                # Pad to square so upscaler works on equal aspect ratio
                side = max(x2 - x1, y2 - y1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                half = side // 2
                x1, y1 = max(0, cx - half), max(0, cy - half)
                x2, y2 = min(w, cx + half), min(h, cy + half)

            cropped = im.crop((x1, y1, x2, y2))

        out_dir  = os.path.dirname(src)
        stem = os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(out_dir, f"_crop_{stem}.png")
        cropped.save(out_path, "PNG")
        log_fn(f"    Smart crop: cropped to {x2-x1}×{y2-y1}px → {os.path.basename(out_path)}", "info")
        return out_path

    except Exception as e:
        log_fn(f"    Smart crop failed: {e}", "error")
        return None


# ── UpscalerApp ───────────────────────────────────────────────────────────────
class UpscalerApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI Image Upscaler")
        self.root.minsize(900, 660)

        icon_path = APP_DIR / "icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception:
                pass

        # State
        self.queue_paths: list[str] = []
        self.running = False
        self._cancel = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._ui_q: queue.Queue = queue.Queue()

        self.cfg = load_settings()

        # Tk variables
        self.var_model     = tk.StringVar(value=self.cfg["model"])
        self.var_model2    = tk.StringVar(value=self.cfg["model2"])
        self.var_scale     = tk.IntVar(value=self.cfg["scale"])
        self.var_tile      = tk.IntVar(value=self.cfg["tile"])
        self.var_threads   = tk.StringVar(value=self.cfg["threads"])
        self.var_outdir    = tk.StringVar(value=self.cfg["output_dir"])
        self.var_format    = tk.StringVar(value=self.cfg["out_format"])
        self.var_jpeg_q    = tk.IntVar(value=self.cfg["jpeg_quality"])
        self.var_webp_q    = tk.IntVar(value=self.cfg["webp_quality"])
        self.var_phase2    = tk.BooleanVar(value=self.cfg["phase2"])
        self.var_dark      = tk.BooleanVar(value=self.cfg["dark_theme"])
        self.var_gpu       = tk.StringVar(value=self.cfg["gpu_id"])
        self.var_tta       = tk.BooleanVar(value=self.cfg["tta_mode"])
        # pre
        self.var_pre_denoise    = tk.BooleanVar(value=self.cfg["pre_denoise"])
        self.var_denoise_h      = tk.IntVar(value=self.cfg["denoise_h"])
        self.var_pre_autolevels = tk.BooleanVar(value=self.cfg["pre_autolevels"])
        self.var_pre_jpeg_fix   = tk.BooleanVar(value=self.cfg["pre_jpeg_fix"])
        # post
        self.var_post_sharpen   = tk.BooleanVar(value=self.cfg["post_sharpen"])
        self.var_sharpen_preset = tk.StringVar(value=self.cfg.get("sharpen_preset", "Photo"))
        self.var_sharpen_radius = tk.DoubleVar(value=self.cfg["sharpen_radius"])
        self.var_sharpen_pct    = tk.IntVar(value=self.cfg["sharpen_pct"])
        self.var_sharpen_thresh = tk.IntVar(value=self.cfg["sharpen_thresh"])
        self.var_post_contrast  = tk.BooleanVar(value=self.cfg["post_contrast"])
        self.var_contrast_f     = tk.DoubleVar(value=self.cfg["contrast_factor"])
        # tools
        self.var_bg_remove      = tk.BooleanVar(value=self.cfg["bg_remove"])
        self.var_smart_crop     = tk.BooleanVar(value=self.cfg["smart_crop"])
        self.var_smart_pad      = tk.BooleanVar(value=self.cfg["smart_pad"])
        self.var_detect_classes = tk.StringVar(value=self.cfg["detect_classes"])
        self.var_preserve_exif  = tk.BooleanVar(value=self.cfg["preserve_exif"])
        self.var_name_template  = tk.StringVar(value=self.cfg["name_template"])

        self.colors = apply_theme(self.root, self.cfg["dark_theme"])
        self._path_before: str | None = None
        self._path_after:  str | None = None
        self._photo_before = None
        self._photo_after  = None

        self._build_ui()
        self._poll_ui_queue()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.var_dark.trace_add("write", lambda *_: self._toggle_theme())

        # Warn on startup if the inference binary is missing — do it after
        # the UI is built so the message appears over the window.
        self.root.after(200, self._check_binary)

    # ═══════════════════════════════════════════════════════════════════
    # UI BUILD
    # ═══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        c = self.colors
        self._build_menu(c)

        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=6, pady=6)
        lf = ttk.Frame(main)
        rf = ttk.Frame(main)
        main.add(lf, weight=2)
        main.add(rf, weight=3)
        self._build_left(lf, c)
        self._build_right(rf, c)

    def _build_menu(self, c):
        mb = tk.Menu(self.root, bg=c["panel"], fg=c["fg"],
                     activebackground=c["accent"], activeforeground="#fff",
                     relief="flat", bd=0)
        fm = tk.Menu(mb, tearoff=0, bg=c["panel"], fg=c["fg"],
                     activebackground=c["accent"], activeforeground="#fff")
        fm.add_command(label="Add Images…",     command=self.add_images)
        fm.add_command(label="Add Folder…",     command=self.add_folder)
        fm.add_separator()
        fm.add_command(label="Open Output Dir", command=self._open_output_dir)
        fm.add_separator()
        fm.add_command(label="Exit",            command=self._on_close)
        mb.add_cascade(label="File", menu=fm)

        tm = tk.Menu(mb, tearoff=0, bg=c["panel"], fg=c["fg"],
                     activebackground=c["accent"], activeforeground="#fff")
        tm.add_command(label="Refresh Model List", command=self._refresh_models)
        tm.add_command(label="Clear Log",          command=self._clear_log)
        mb.add_cascade(label="Tools", menu=tm)

        hm = tk.Menu(mb, tearoff=0, bg=c["panel"], fg=c["fg"],
                     activebackground=c["accent"], activeforeground="#fff")
        hm.add_command(label="Help — User Guide",        command=self.show_help)
        hm.add_command(label="Model Reference",          command=self.show_model_guide)
        hm.add_separator()
        hm.add_command(label="Real-ESRGAN GitHub",
                       command=lambda: webbrowser.open("https://github.com/xinntao/Real-ESRGAN"))
        hm.add_command(label="Model Browser (OpenModelDB)",
                       command=lambda: webbrowser.open("https://openmodeldb.info"))
        hm.add_separator()
        hm.add_command(label="About",                    command=self.show_about)
        mb.add_cascade(label="Help", menu=hm)
        self.root.config(menu=mb)

    def _build_left(self, parent, c):
        # toolbar
        tb = ttk.Frame(parent); tb.pack(fill="x", pady=(0, 4))
        ttk.Button(tb, text="+ Images",  command=self.add_images).pack(side="left", padx=2)
        ttk.Button(tb, text="+ Folder",  command=self.add_folder).pack(side="left", padx=2)
        ttk.Button(tb, text="Remove",    command=self.remove_selected).pack(side="left", padx=2)
        ttk.Button(tb, text="Clear All", command=self.clear_queue).pack(side="left", padx=2)

        # queue listbox
        qf = ttk.LabelFrame(parent, text="Queue")
        qf.pack(fill="both", expand=True)
        vsb = ttk.Scrollbar(qf, orient="vertical")
        hsb = ttk.Scrollbar(qf, orient="horizontal")
        self.queue_box = tk.Listbox(
            qf, selectmode="extended",
            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
            bg=c["entry"], fg=c["fg"], selectbackground=c["select"],
            selectforeground=c["fg"], bd=0, highlightthickness=0,
            activestyle="none", font=("Segoe UI", 9),
        )
        vsb.config(command=self.queue_box.yview)
        hsb.config(command=self.queue_box.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.queue_box.pack(fill="both", expand=True)
        self.queue_box.bind("<Double-Button-1>", lambda _: self.remove_selected())
        self.queue_box.bind("<Delete>",          lambda _: self.remove_selected())
        if DND_AVAILABLE:
            self.queue_box.drop_target_register(DND_FILES)
            self.queue_box.dnd_bind("<<Drop>>", self._on_drop)
            qf.config(text="Queue  (drag & drop enabled)")

        # reorder
        order = ttk.Frame(parent); order.pack(fill="x", pady=2)
        ttk.Button(order, text="▲ Up",   command=self.move_up).pack(side="left", padx=2)
        ttk.Button(order, text="▼ Down", command=self.move_down).pack(side="left", padx=2)
        self.lbl_count = ttk.Label(order, text="0 items")
        self.lbl_count.pack(side="right", padx=4)

        # settings notebook
        nb = ttk.Notebook(parent); nb.pack(fill="x", pady=(4, 0))
        self._build_tab_model(nb, c)
        self._build_tab_prepost(nb, c)
        self._build_tab_tools(nb, c)
        self._build_tab_output(nb, c)
        self._build_tab_advanced(nb, c)

        # progress + status
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill="x", pady=(6, 2))
        self.lbl_status = ttk.Label(parent, text="Idle", anchor="w")
        self.lbl_status.pack(fill="x")

        # start / cancel
        btns = ttk.Frame(parent); btns.pack(fill="x", pady=4)
        self.btn_start  = ttk.Button(btns, text="▶  Start",
                                     style="Accent.TButton", command=self.start)
        self.btn_cancel = ttk.Button(btns, text="■  Cancel",
                                     command=self.cancel, state="disabled")
        self.btn_start.pack(side="left",  padx=2, fill="x", expand=True)
        self.btn_cancel.pack(side="left", padx=2, fill="x", expand=True)

    # ── Settings tabs ─────────────────────────────────────────────────
    def _row(self, parent, label: str, width: int = 14):
        r = ttk.Frame(parent); r.pack(fill="x", padx=6, pady=2)
        if label:
            ttk.Label(r, text=label, width=width, anchor="w").pack(side="left")
        return r

    def _build_tab_model(self, nb, c):
        tab = ttk.Frame(nb); nb.add(tab, text="Model")

        r1 = self._row(tab, "Model:")
        self.model_box = ttk.Combobox(r1, textvariable=self.var_model,
                                      values=scan_models(), state="readonly", width=26)
        self.model_box.pack(side="left", fill="x", expand=True, padx=4)
        self.model_box.bind("<<ComboboxSelected>>", self._on_model_change)
        ttk.Button(r1, text="ℹ", width=3, command=self._show_model_info).pack(side="left")

        r2 = self._row(tab, "Scale:")
        self.scale_box = ttk.Combobox(r2, textvariable=self.var_scale,
                                      values=[2, 3, 4], state="readonly", width=6)
        self.scale_box.pack(side="left", padx=4)

        r3 = self._row(tab, "")
        ttk.Checkbutton(r3, text="TTA mode (slower, more precise)",
                        variable=self.var_tta).pack(side="left")

        r4 = self._row(tab, "")
        ttk.Checkbutton(r4, text="2-Pass Enhance", variable=self.var_phase2,
                        command=self._toggle_phase2).pack(side="left")

        self.frame_model2 = ttk.Frame(tab)
        self.frame_model2.pack(fill="x", padx=6, pady=(0, 2))
        ttk.Label(self.frame_model2, text="Pass-2 Model:", width=14, anchor="w").pack(side="left")
        self.model2_box = ttk.Combobox(self.frame_model2, textvariable=self.var_model2,
                                       values=scan_models(), state="readonly", width=26)
        self.model2_box.pack(side="left", fill="x", expand=True, padx=4)

        self._toggle_phase2()
        self._on_model_change()

    def _build_tab_prepost(self, nb, c):
        tab = ttk.Frame(nb); nb.add(tab, text="Pre/Post")

        pre = ttk.LabelFrame(tab, text="Pre-processing  (applied before upscale)")
        pre.pack(fill="x", padx=4, pady=4)

        r1 = self._row(pre, "")
        ttk.Checkbutton(r1, text="Auto-levels (normalise contrast)",
                        variable=self.var_pre_autolevels).pack(side="left")

        r2 = self._row(pre, "")
        ttk.Checkbutton(r2, text="JPEG artifact softening",
                        variable=self.var_pre_jpeg_fix).pack(side="left")

        r3 = self._row(pre, "")
        ttk.Checkbutton(r3, text="NLM Denoise  h=",
                        variable=self.var_pre_denoise).pack(side="left")
        ttk.Spinbox(r3, textvariable=self.var_denoise_h,
                    from_=1, to=30, width=4).pack(side="left")
        ttk.Label(r3, text="(1=light  10=medium  20=heavy)",
                  foreground=c["border"]).pack(side="left", padx=4)
        if not CV2_AVAILABLE:
            ttk.Label(r3, text="⚠ opencv not installed",
                      foreground=c["warn"]).pack(side="left")

        post = ttk.LabelFrame(tab, text="Post-processing  (applied after upscale)")
        post.pack(fill="x", padx=4, pady=4)

        # Sharpen enable + preset selector on one row
        rp = self._row(post, "")
        ttk.Checkbutton(rp, text="Sharpen",
                        variable=self.var_post_sharpen).pack(side="left")
        ttk.Label(rp, text="  Preset:", foreground=c["border"]).pack(side="left")
        preset_box = ttk.Combobox(rp, textvariable=self.var_sharpen_preset,
                                  values=list(SHARPEN_PRESETS.keys()),
                                  state="readonly", width=8)
        preset_box.pack(side="left", padx=4)
        preset_box.bind("<<ComboboxSelected>>", self._on_sharpen_preset)

        # Custom spinboxes (hidden when a named preset is active)
        self.frame_sharpen_custom = ttk.Frame(post)
        self.frame_sharpen_custom.pack(fill="x", padx=6, pady=2)
        ttk.Label(self.frame_sharpen_custom, text="radius=", foreground=c["border"]).pack(side="left")
        ttk.Spinbox(self.frame_sharpen_custom, textvariable=self.var_sharpen_radius,
                    from_=0.5, to=5.0, increment=0.5, width=5, format="%.1f",
                    command=self._on_sharpen_custom).pack(side="left")
        ttk.Label(self.frame_sharpen_custom, text="  pct=", foreground=c["border"]).pack(side="left")
        ttk.Spinbox(self.frame_sharpen_custom, textvariable=self.var_sharpen_pct,
                    from_=50, to=300, width=5,
                    command=self._on_sharpen_custom).pack(side="left")
        ttk.Label(self.frame_sharpen_custom, text="  thresh=", foreground=c["border"]).pack(side="left")
        ttk.Spinbox(self.frame_sharpen_custom, textvariable=self.var_sharpen_thresh,
                    from_=0, to=10, width=4,
                    command=self._on_sharpen_custom).pack(side="left")

        # Sharpen-Only button — applies post-processing to already-upscaled files in queue
        rso = self._row(post, "")
        ttk.Button(rso, text="✦ Sharpen Only",
                   command=self.sharpen_only).pack(side="left", padx=2)
        ttk.Label(rso,
                  text="Apply sharpen/contrast to queued images without re-upscaling",
                  foreground=c["border"]).pack(side="left", padx=6)

        rc = self._row(post, "")
        ttk.Checkbutton(rc, text="Contrast enhance  factor=",
                        variable=self.var_post_contrast).pack(side="left")
        ttk.Spinbox(rc, textvariable=self.var_contrast_f,
                    from_=0.8, to=2.0, increment=0.05, width=5,
                    format="%.2f").pack(side="left")
        ttk.Label(rc, text="(1.0 = no change)",
                  foreground=c["border"]).pack(side="left", padx=4)

        # Initialise custom frame visibility
        self._on_sharpen_preset()

    def _build_tab_tools(self, nb, c):
        tab = ttk.Frame(nb); nb.add(tab, text="Tools")

        bg_f = ttk.LabelFrame(tab, text="Background Removal  (rembg / U2Net)")
        bg_f.pack(fill="x", padx=4, pady=4)

        r1 = self._row(bg_f, "")
        ttk.Checkbutton(r1, text="Remove background (saves separate _nobg.png)",
                        variable=self.var_bg_remove).pack(side="left")
        if not REMBG_AVAILABLE:
            ttk.Label(r1, text="⚠ install: pip install 'rembg[cpu]'",
                      foreground=c["warn"]).pack(side="left", padx=6)

        sc_f = ttk.LabelFrame(tab, text="Smart Crop / Subject Focus  (YOLOv8n)")
        sc_f.pack(fill="x", padx=4, pady=4)

        r2 = self._row(sc_f, "")
        ttk.Checkbutton(r2, text="Auto-crop to detected subject before upscale",
                        variable=self.var_smart_crop).pack(side="left")
        if YOLO_AVAILABLE is False:
            ttk.Label(r2, text="⚠ install: pip install ultralytics",
                      foreground=c["warn"]).pack(side="left", padx=6)

        r3 = self._row(sc_f, "Detect classes:")
        ttk.Entry(r3, textvariable=self.var_detect_classes, width=28).pack(side="left", padx=4)
        ttk.Label(r3, text="comma-separated  e.g. person,animal,car",
                  foreground=c["border"]).pack(side="left")

        r4 = self._row(sc_f, "")
        ttk.Checkbutton(r4, text="Pad crop to square (recommended for upscaling)",
                        variable=self.var_smart_pad).pack(side="left")

        meta_f = ttk.LabelFrame(tab, text="Metadata")
        meta_f.pack(fill="x", padx=4, pady=4)
        r5 = self._row(meta_f, "")
        ttk.Checkbutton(r5, text="Preserve EXIF metadata in output (JPEG only)",
                        variable=self.var_preserve_exif).pack(side="left")
        if not PIEXIF_AVAILABLE:
            ttk.Label(r5, text="⚠ install: pip install piexif",
                      foreground=c["warn"]).pack(side="left", padx=6)

    def _build_tab_output(self, nb, c):
        tab = ttk.Frame(nb); nb.add(tab, text="Output")

        r1 = self._row(tab, "Save to:", 10)
        ttk.Entry(r1, textvariable=self.var_outdir).pack(side="left", fill="x",
                                                          expand=True, padx=4)
        ttk.Button(r1, text="…", width=3, command=self._pick_outdir).pack(side="left")

        r2 = self._row(tab, "Format:", 10)
        fb = ttk.Combobox(r2, textvariable=self.var_format,
                          values=list(OUTPUT_FORMATS), state="readonly", width=8)
        fb.pack(side="left", padx=4)
        fb.bind("<<ComboboxSelected>>", self._on_format_change)

        self.frame_jpeg = ttk.Frame(tab)
        self.frame_jpeg.pack(fill="x", padx=6, pady=2)
        ttk.Label(self.frame_jpeg, text="JPEG quality:", width=14, anchor="w").pack(side="left")
        ttk.Scale(self.frame_jpeg, from_=10, to=100, orient="horizontal",
                  variable=self.var_jpeg_q, length=120).pack(side="left", padx=4)
        self.lbl_jpeg_q = ttk.Label(self.frame_jpeg, text=str(self.var_jpeg_q.get()))
        self.lbl_jpeg_q.pack(side="left")
        self.var_jpeg_q.trace_add("write",
            lambda *_: self.lbl_jpeg_q.config(text=str(self.var_jpeg_q.get())))

        self.frame_webp = ttk.Frame(tab)
        self.frame_webp.pack(fill="x", padx=6, pady=2)
        ttk.Label(self.frame_webp, text="WebP quality:", width=14, anchor="w").pack(side="left")
        ttk.Scale(self.frame_webp, from_=10, to=100, orient="horizontal",
                  variable=self.var_webp_q, length=120).pack(side="left", padx=4)
        self.lbl_webp_q = ttk.Label(self.frame_webp, text=str(self.var_webp_q.get()))
        self.lbl_webp_q.pack(side="left")
        self.var_webp_q.trace_add("write",
            lambda *_: self.lbl_webp_q.config(text=str(self.var_webp_q.get())))

        r3 = self._row(tab, "Filename:", 10)
        ttk.Entry(r3, textvariable=self.var_name_template,
                  width=30).pack(side="left", padx=4)
        ttk.Label(r3, text="  {stem} {model} {scale} {ext}",
                  foreground=c["border"]).pack(side="left")

        self._on_format_change()

    def _build_tab_advanced(self, nb, c):
        tab = ttk.Frame(nb); nb.add(tab, text="Advanced")

        r1 = self._row(tab, "GPU ID:")
        ttk.Entry(r1, textvariable=self.var_gpu, width=8).pack(side="left", padx=4)
        ttk.Label(r1, text="0, 1, … or auto", foreground=c["border"]).pack(side="left")

        r2 = self._row(tab, "Tile size:")
        ttk.Entry(r2, textvariable=self.var_tile, width=8).pack(side="left", padx=4)
        ttk.Label(r2, text="(0 = auto, try 128/256 if VRAM limited)",
                  foreground=c["border"]).pack(side="left")

        r3 = self._row(tab, "Threads:")
        ttk.Entry(r3, textvariable=self.var_threads, width=10).pack(side="left", padx=4)
        ttk.Label(r3, text="load:proc:save  e.g. 2:2:2",
                  foreground=c["border"]).pack(side="left")

        r4 = self._row(tab, "")
        ttk.Checkbutton(r4, text="Dark theme", variable=self.var_dark).pack(side="left")

    def _build_right(self, parent, c):
        rp = ttk.PanedWindow(parent, orient="vertical")
        rp.pack(fill="both", expand=True)

        pf = ttk.LabelFrame(rp, text="Preview")
        lf = ttk.LabelFrame(rp, text="Log")
        # Also add model reference panel
        mf = ttk.LabelFrame(rp, text="Model Reference")
        rp.add(pf, weight=3)
        rp.add(lf, weight=1)
        rp.add(mf, weight=1)

        # Preview
        pf.columnconfigure(0, weight=1)
        pf.columnconfigure(1, weight=1)
        pf.rowconfigure(1, weight=1)
        ttk.Label(pf, text="Before", foreground=c["accent"]).grid(row=0, column=0, pady=(4, 0))
        ttk.Label(pf, text="After",  foreground=c["success"]).grid(row=0, column=1, pady=(4, 0))
        self.lbl_before_info = ttk.Label(pf, text="—",
                                         foreground=c["border"], font=("Segoe UI", 8))
        self.lbl_before_info.grid(row=2, column=0, sticky="ew", padx=4)
        self.lbl_after_info  = ttk.Label(pf, text="—",
                                         foreground=c["border"], font=("Segoe UI", 8))
        self.lbl_after_info.grid(row=2, column=1, sticky="ew", padx=4)
        self.canvas_before = tk.Canvas(pf, bg=c["entry"], bd=0, highlightthickness=0)
        self.canvas_after  = tk.Canvas(pf, bg=c["entry"], bd=0, highlightthickness=0)
        self.canvas_before.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        self.canvas_after.grid( row=1, column=1, sticky="nsew", padx=4, pady=4)
        self.canvas_before.bind("<Configure>", lambda _: self._redraw_canvas("before"))
        self.canvas_after.bind( "<Configure>", lambda _: self._redraw_canvas("after"))

        # Log
        log_vsb = ttk.Scrollbar(lf, orient="vertical")
        self.log_text = tk.Text(
            lf, height=6, wrap="word",
            bg=c["entry"], fg=c["fg"], insertbackground=c["fg"],
            bd=0, highlightthickness=0, font=("Consolas", 8),
            yscrollcommand=log_vsb.set, state="disabled",
        )
        log_vsb.config(command=self.log_text.yview)
        log_vsb.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True, padx=4, pady=4)
        for tag, colour in [("info", c["fg"]), ("success", c["success"]),
                             ("warn", c["warn"]), ("error", c["error"])]:
            self.log_text.tag_configure(tag, foreground=colour)

        # Model reference
        ref_vsb = ttk.Scrollbar(mf, orient="vertical")
        self.ref_text = tk.Text(
            mf, height=5, wrap="word",
            bg=c["entry"], fg=c["fg"], insertbackground=c["fg"],
            bd=0, highlightthickness=0, font=("Segoe UI", 8),
            yscrollcommand=ref_vsb.set, state="disabled",
        )
        ref_vsb.config(command=self.ref_text.yview)
        ref_vsb.pack(side="right", fill="y")
        self.ref_text.pack(fill="both", expand=True, padx=4, pady=4)
        self.ref_text.tag_configure("heading", foreground=c["accent"],
                                    font=("Segoe UI", 9, "bold"))
        self.ref_text.tag_configure("body",    foreground=c["fg"])
        self._populate_model_reference()

    def _populate_model_reference(self):
        self.ref_text.config(state="normal")
        self.ref_text.delete("1.0", "end")
        for model, desc in sorted(MODEL_DESCRIPTIONS.items()):
            self.ref_text.insert("end", f"{model}\n", "heading")
            self.ref_text.insert("end", f"  {desc}\n\n", "body")
        self.ref_text.config(state="disabled")

    # ═══════════════════════════════════════════════════════════════════
    # UI HELPERS
    # ═══════════════════════════════════════════════════════════════════
    def _toggle_phase2(self):
        if self.var_phase2.get():
            self.frame_model2.pack(fill="x", padx=6, pady=(0, 2))
        else:
            self.frame_model2.pack_forget()

    def _on_model_change(self, _=None):
        m = self.var_model.get()
        if m in FIXED_SCALE_MODELS:
            self.var_scale.set(FIXED_SCALE_MODELS[m])
            self.scale_box.config(state="disabled")
        else:
            self.scale_box.config(state="readonly")

    def _on_format_change(self, _=None):
        fmt = self.var_format.get()
        if fmt == "jpg":
            self.frame_jpeg.pack(fill="x", padx=6, pady=2)
            self.frame_webp.pack_forget()
        elif fmt == "webp":
            self.frame_webp.pack(fill="x", padx=6, pady=2)
            self.frame_jpeg.pack_forget()
        else:
            self.frame_jpeg.pack_forget()
            self.frame_webp.pack_forget()

    def _pick_outdir(self):
        d = filedialog.askdirectory(initialdir=self.var_outdir.get())
        if d:
            self.var_outdir.set(d)

    def _open_output_dir(self):
        d = self.var_outdir.get()
        os.makedirs(d, exist_ok=True)
        if platform.system() == "Windows":
            os.startfile(d)
        else:
            subprocess.Popen(["xdg-open", d])

    def _toggle_theme(self):
        self.colors = apply_theme(self.root, self.var_dark.get())

    def _refresh_models(self):
        m = scan_models()
        self.model_box["values"]  = m
        self.model2_box["values"] = m
        self._log(f"Found {len(m)} model(s).", "info")

    def _check_binary(self):
        """Called 200 ms after startup. Warns if the inference binary is missing."""
        binary = APP_DIR / EXE_NAME
        if not binary.exists():
            missing = str(binary)
            msg = (
                f"Inference binary not found:\n{missing}\n\n"
                f"The file  realesrgan-ncnn-vulkan.exe  must be in the same\n"
                f"folder as this application.\n\n"
                f"If you are running the portable EXE, make sure you placed it\n"
                f"alongside the models/ folder from the repository."
            )
            messagebox.showerror("Missing Binary", msg)
        if not MODELS_DIR.exists() or not any(MODELS_DIR.glob("*.param")):
            msg = (
                f"No NCNN models found in:\n{MODELS_DIR}\n\n"
                f"The models/ folder must be in the same folder as the EXE.\n"
                f"Download the full release or clone the repository."
            )
            messagebox.showwarning("No Models Found", msg)

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _update_count(self):
        n = len(self.queue_paths)
        self.lbl_count.config(text=f"{n} item{'s' if n != 1 else ''}")

    def _show_model_info(self):
        m = self.var_model.get()
        desc = MODEL_DESCRIPTIONS.get(m, DEFAULT_MODEL_DESC)
        messagebox.showinfo(f"Model: {m}", desc)

    def _on_sharpen_preset(self, _=None):
        """When a named preset is chosen, load its values and hide/show custom spinboxes."""
        name = self.var_sharpen_preset.get()
        values = SHARPEN_PRESETS.get(name)
        if values is not None:
            r, p, t = values
            self.var_sharpen_radius.set(r)
            self.var_sharpen_pct.set(p)
            self.var_sharpen_thresh.set(t)
            self.frame_sharpen_custom.pack_forget()
        else:
            # "Custom" — show spinboxes beneath the preset row
            self.frame_sharpen_custom.pack(fill="x", padx=6, pady=2)

    def _on_sharpen_custom(self):
        """If the user manually tweaks a spinbox, switch preset to Custom."""
        self.var_sharpen_preset.set("Custom")
        self.frame_sharpen_custom.pack(fill="x", padx=6, pady=2)

    def sharpen_only(self):
        """
        Apply post-processing (sharpen + contrast) to every queued file
        in-place without running the upscale binary.  Useful for re-sharpening
        already-upscaled images that came out too soft.
        """
        if self.running:
            messagebox.showwarning("Busy", "Wait for the current batch to finish.")
            return
        if not self.queue_paths:
            messagebox.showinfo("Nothing to do", "Add images to the queue first.")
            return
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow missing", "Pillow is required for sharpening.")
            return

        cfg      = self._collect_cfg()
        snapshot = list(self.queue_paths)
        self.running = True
        self._cancel.clear()
        self.progress["value"] = 0
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self._clear_log()
        self._log(f"Sharpen-Only: processing {len(snapshot)} image(s)…", "info")
        preset = cfg.get("sharpen_preset", "Custom")
        self._log(f"  Preset={preset}  r={cfg['sharpen_radius']}  "
                  f"pct={cfg['sharpen_pct']}%  thresh={cfg['sharpen_thresh']}", "info")

        def _run():
            ok = fail = 0
            for i, src in enumerate(snapshot):
                if self._cancel.is_set():
                    self._log("Cancelled.", "warn")
                    break
                self._set_status(f"Sharpening {i+1}/{len(snapshot)}: {os.path.basename(src)}")
                self._show_before(src)
                try:
                    # Write to output dir so we don't overwrite originals
                    out_dir = cfg["output_dir"]
                    os.makedirs(out_dir, exist_ok=True)
                    stem = os.path.splitext(os.path.basename(src))[0]
                    ext  = os.path.splitext(src)[1] or ".png"
                    out  = os.path.join(out_dir, f"{stem}_sharp{ext}")
                    import shutil
                    shutil.copy2(src, out)
                    postprocess_image(out, cfg, self._log)
                    self._show_after(out)
                    self._log(f"  ✓ → {os.path.basename(out)}", "success")
                    ok += 1
                except Exception as e:
                    self._log(f"  FAILED: {e}", "error")
                    fail += 1
                self._set_progress((i + 1) / len(snapshot) * 100)
            self._log(f"\nDone: {ok} sharpened, {fail} failed.", "success" if fail == 0 else "warn")
            self._ui_q.put(("done",))

        threading.Thread(target=_run, daemon=True).start()

    # ── thread-safe UI pump ───────────────────────────────────────────
    def _log(self, msg, tag="info"):
        self._ui_q.put(("log", msg, tag))

    def _set_status(self, text):
        self._ui_q.put(("status", text))

    def _set_progress(self, val):
        self._ui_q.put(("progress", val))

    def _show_before(self, path):
        self._ui_q.put(("before", path))

    def _show_after(self, path):
        self._ui_q.put(("after", path))

    def _poll_ui_queue(self):
        try:
            while True:
                self._dispatch(self._ui_q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(50, self._poll_ui_queue)

    def _dispatch(self, msg):
        k = msg[0]
        if k == "log":
            self.log_text.config(state="normal")
            self.log_text.insert("end", msg[1] + "\n", msg[2])
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        elif k == "status":
            self.lbl_status.config(text=msg[1])
        elif k == "progress":
            self.progress["value"] = msg[1]
        elif k == "before":
            self._path_before = msg[1]
            self._render_canvas(self.canvas_before, msg[1], "before")
        elif k == "after":
            self._path_after = msg[1]
            self._render_canvas(self.canvas_after, msg[1], "after")
        elif k == "done":
            self._on_worker_done()

    def _redraw_canvas(self, side):
        path = self._path_before if side == "before" else self._path_after
        canvas = self.canvas_before if side == "before" else self.canvas_after
        if path:
            self._render_canvas(canvas, path, side)

    def _render_canvas(self, canvas, path, side):
        if not PIL_AVAILABLE:
            return
        try:
            w = canvas.winfo_width()  or 320
            h = canvas.winfo_height() or 280
            with Image.open(path) as im:
                im.thumbnail((max(w, 100), max(h, 100)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(im)
            canvas.delete("all")
            canvas.create_image(w // 2, h // 2, anchor="center", image=photo)
            if side == "before":
                self._photo_before = photo
                self.lbl_before_info.config(text=image_info(path))
            else:
                self._photo_after  = photo
                self.lbl_after_info.config(text=image_info(path))
        except Exception as e:
            self._log(f"Preview error: {e}", "warn")

    # ═══════════════════════════════════════════════════════════════════
    # QUEUE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════
    def _add_paths(self, paths: list[str]):
        added = 0
        for p in paths:
            if is_image(p) and p not in self.queue_paths:
                self.queue_paths.append(p)
                self.queue_box.insert("end", os.path.basename(p))
                added += 1
        self._update_count()
        if added:
            self._log(f"Added {added} image(s).", "info")

    def add_images(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Images", " ".join(f"*{e}" for e in SUPPORTED_EXTS)),
                       ("All files", "*.*")])
        if files:
            self._add_paths(list(files))

    def add_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        paths = [os.path.join(r, f)
                 for r, _, fs in os.walk(folder) for f in fs]
        self._add_paths(paths)

    def remove_selected(self):
        for i in reversed(list(self.queue_box.curselection())):
            self.queue_box.delete(i)
            del self.queue_paths[i]
        self._update_count()

    def clear_queue(self):
        self.queue_paths.clear()
        self.queue_box.delete(0, "end")
        self._update_count()

    def move_up(self):
        sel = self.queue_box.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        self.queue_paths[i], self.queue_paths[i-1] = \
            self.queue_paths[i-1], self.queue_paths[i]
        self._refresh_queue(i - 1)

    def move_down(self):
        sel = self.queue_box.curselection()
        if not sel or sel[0] >= len(self.queue_paths) - 1:
            return
        i = sel[0]
        self.queue_paths[i], self.queue_paths[i+1] = \
            self.queue_paths[i+1], self.queue_paths[i]
        self._refresh_queue(i + 1)

    def _refresh_queue(self, sel: int | None = None):
        self.queue_box.delete(0, "end")
        for p in self.queue_paths:
            self.queue_box.insert("end", os.path.basename(p))
        if sel is not None:
            self.queue_box.selection_set(sel)
        self._update_count()

    def _on_drop(self, event):
        try:
            paths = list(self.root.tk.splitlist(event.data))
        except Exception:
            paths = event.data.split()
        expanded: list[str] = []
        for p in paths:
            if os.path.isdir(p):
                for r, _, fs in os.walk(p):
                    for f in fs:
                        expanded.append(os.path.join(r, f))
            else:
                expanded.append(p)
        self._add_paths(expanded)

    # ═══════════════════════════════════════════════════════════════════
    # CORE UPSCALING
    # ═══════════════════════════════════════════════════════════════════
    def _collect_cfg(self) -> dict:
        """Snapshot current UI settings into a plain dict for the worker thread."""
        return {
            "model":          self.var_model.get(),
            "model2":         self.var_model2.get(),
            "scale":          self.var_scale.get(),
            "tile":           self.var_tile.get(),
            "threads":        self.var_threads.get(),
            "gpu_id":         self.var_gpu.get(),
            "tta_mode":       self.var_tta.get(),
            "phase2":         self.var_phase2.get(),
            "output_dir":     self.var_outdir.get(),
            "out_format":     self.var_format.get(),
            "jpeg_quality":   self.var_jpeg_q.get(),
            "webp_quality":   self.var_webp_q.get(),
            "name_template":  self.var_name_template.get(),
            "preserve_exif":  self.var_preserve_exif.get(),
            # pre
            "pre_denoise":    self.var_pre_denoise.get(),
            "denoise_h":      self.var_denoise_h.get(),
            "pre_autolevels": self.var_pre_autolevels.get(),
            "pre_jpeg_fix":   self.var_pre_jpeg_fix.get(),
            # post
            "post_sharpen":   self.var_post_sharpen.get(),
            "sharpen_preset": self.var_sharpen_preset.get(),
            "sharpen_radius": self.var_sharpen_radius.get(),
            "sharpen_pct":    self.var_sharpen_pct.get(),
            "sharpen_thresh": self.var_sharpen_thresh.get(),
            "post_contrast":  self.var_post_contrast.get(),
            "contrast_factor":self.var_contrast_f.get(),
            # tools
            "bg_remove":      self.var_bg_remove.get(),
            "smart_crop":     self.var_smart_crop.get(),
            "smart_pad":      self.var_smart_pad.get(),
            "detect_classes": self.var_detect_classes.get(),
        }

    def _build_cmd(self, inp: str, out: str, model: str, cfg: dict) -> list[str]:
        scale = FIXED_SCALE_MODELS.get(model, cfg["scale"])
        ncnn_model = model
        if model.startswith("realesr-animevideov3-x"):
            ncnn_model = "realesr-animevideov3"
        cmd = [
            str(APP_DIR / EXE_NAME),
            "-i", inp,
            "-o", out,
            "-n", ncnn_model,
            "-s", str(scale),
            "-g", str(cfg["gpu_id"]),
            "-j", cfg["threads"],
            "-t", str(cfg["tile"]),
            "-m", str(MODELS_DIR),   # ← FIXED: always pass model dir
        ]
        if cfg["tta_mode"]:
            cmd.append("-x")
        return cmd

    def _get_out_png(self, inp: str, suffix: str, cfg: dict) -> str:
        out_dir = cfg["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        base  = os.path.basename(inp)
        stem  = os.path.splitext(base)[0]
        ext_s = os.path.splitext(base)[1].lstrip(".").lower() or "img"
        return os.path.join(out_dir, f"{stem}_{ext_s}{suffix}.png")

    def _run_binary(self, inp: str, model: str, suffix: str, cfg: dict) -> str | None:
        out_png = self._get_out_png(inp, suffix, cfg)
        cmd     = self._build_cmd(inp, out_png, model, cfg)
        self._log(f"  cmd: {' '.join(cmd)}", "info")
        try:
            flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=flags,
            )
            stdout, _ = self._proc.communicate()
            for line in (stdout or "").splitlines():
                line = line.rstrip()
                if line:
                    tag = "error" if "error" in line.lower() else "info"
                    self._log(f"    {line}", tag)
            rc = self._proc.returncode
            self._proc = None
            if self._cancel.is_set():
                return None
            if rc not in (0,):
                self._log(f"  Binary exit code {rc}.", "error")
                return None
            if not os.path.exists(out_png):
                self._log("  Output file not created.", "error")
                return None
            return out_png
        except FileNotFoundError:
            self._log(f"  Executable not found: {APP_DIR / EXE_NAME}", "error")
            return None
        except Exception as e:
            self._log(f"  Subprocess error: {e}", "error")
            return None

    def _convert_format(self, png_path: str, cfg: dict, orig_src: str) -> str:
        fmt = cfg["out_format"]
        stem_in   = os.path.splitext(os.path.basename(png_path))[0]
        # Apply the user's filename template
        model     = cfg["model"].replace("/", "_")
        scale     = FIXED_SCALE_MODELS.get(cfg["model"], cfg["scale"])
        orig_stem = os.path.splitext(os.path.basename(orig_src))[0]
        final_name = format_name(cfg["name_template"], orig_stem, model, scale, fmt)
        out_path   = os.path.join(cfg["output_dir"], f"{final_name}.{fmt}")

        if fmt == "png" and not cfg.get("post_sharpen") and not cfg.get("post_contrast"):
            # Just rename/move the intermediate file
            if png_path != out_path:
                try:
                    os.replace(png_path, out_path)
                except OSError:
                    import shutil
                    shutil.copy2(png_path, out_path)
                    os.remove(png_path)
            return out_path

        if not PIL_AVAILABLE:
            return png_path

        try:
            with Image.open(png_path) as im:
                save_im = im.copy()

            if fmt == "jpg":
                if save_im.mode in ("RGBA", "LA", "P"):
                    save_im = save_im.convert("RGB")
                save_im.save(out_path, "JPEG", quality=cfg["jpeg_quality"])
                if cfg["preserve_exif"]:
                    copy_exif(orig_src, out_path)
            elif fmt == "webp":
                save_im.save(out_path, "WEBP", quality=cfg["webp_quality"], method=6)
            else:
                save_im.save(out_path, "PNG")

            if png_path != out_path and os.path.exists(png_path):
                os.remove(png_path)
            return out_path
        except Exception as e:
            self._log(f"  Format conversion error: {e}", "warn")
            return png_path

    # ═══════════════════════════════════════════════════════════════════
    # WORKER THREAD
    # ═══════════════════════════════════════════════════════════════════
    def _worker(self, snapshot: list[str], cfg: dict):
        total = len(snapshot)
        ok = fail = 0
        t_session = time.time()
        timings: list[float] = []
        tmp_files: list[str] = []   # track temp pre-processing files for cleanup

        for i, orig in enumerate(snapshot):
            if self._cancel.is_set():
                self._log("Cancelled.", "warn")
                break

            t0 = time.time()
            self._set_status(f"Processing {i+1}/{total}: {os.path.basename(orig)}")
            self._log(f"\n[{i+1}/{total}] {orig}", "info")
            self._show_before(orig)

            # ── Smart crop ───────────────────────────────────────────
            work = orig
            if cfg["smart_crop"]:
                cropped = detect_and_crop(orig, cfg, self._log)
                if cropped:
                    work = cropped
                    tmp_files.append(cropped)

            # ── Pre-processing ───────────────────────────────────────
            pre_path = preprocess_image(work, cfg, self._log)
            if pre_path != work:
                tmp_files.append(pre_path)
            work = pre_path

            # ── Pass 1 upscale ───────────────────────────────────────
            out = self._run_binary(work, cfg["model"], "_up", cfg)
            if out is None:
                fail += 1
                self._log("  FAILED.", "error")
                self._set_progress((i + 1) / total * 100)
                continue

            # ── Pass 2 upscale (optional) ────────────────────────────
            if cfg["phase2"] and not self._cancel.is_set():
                self._log(f"  Pass 2 → {cfg['model2']}…", "info")
                out2 = self._run_binary(out, cfg["model2"], "_up2", cfg)
                if out2:
                    try:
                        os.remove(out)
                    except OSError:
                        pass
                    out = out2
                else:
                    self._log("  Pass 2 failed — keeping pass-1 output.", "warn")

            # ── Post-processing ──────────────────────────────────────
            postprocess_image(out, cfg, self._log)

            # ── Format conversion + final filename ───────────────────
            final = self._convert_format(out, cfg, orig)

            # ── Background removal (separate output) ─────────────────
            if cfg["bg_remove"]:
                remove_background(
                    final,
                    cfg["output_dir"],
                    os.path.splitext(os.path.basename(final))[0],
                    self._log,
                )

            elapsed = time.time() - t0
            timings.append(elapsed)
            ok += 1
            self._show_after(final)
            self._set_progress((i + 1) / total * 100)

            rem = total - (i + 1)
            eta = ""
            if rem > 0 and timings:
                avg = sum(timings) / len(timings)
                s   = int(avg * rem)
                eta = f"  ETA ~{s//60}m{s%60:02d}s" if s >= 60 else f"  ETA ~{s}s"
            self._set_status(
                f"Done {i+1}/{total} in {elapsed:.1f}s — {os.path.basename(final)}{eta}")
            self._log(f"  ✓ {elapsed:.1f}s → {final}", "success")

        # Cleanup temp pre-processing files
        for f in tmp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass

        total_t = time.time() - t_session
        self._log(
            f"\nBatch complete: {ok} succeeded, {fail} failed in {total_t:.1f}s",
            "success" if fail == 0 else "warn",
        )
        self._ui_q.put(("done",))

    def _on_worker_done(self):
        self.running = False
        self._cancel.clear()
        self.btn_start.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self.lbl_status.config(text="Finished — ready.")

    # ═══════════════════════════════════════════════════════════════════
    # START / CANCEL
    # ═══════════════════════════════════════════════════════════════════
    def start(self):
        if self.running:
            return
        if not self.queue_paths:
            messagebox.showinfo("Nothing to do", "Add images to the queue first.")
            return
        self.running = True
        self._cancel.clear()
        self.progress["value"] = 0
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self._clear_log()
        cfg      = self._collect_cfg()
        snapshot = list(self.queue_paths)
        self._log(f"Starting batch of {len(snapshot)} image(s)…", "info")
        self._log(f"  Model: {cfg['model']}  Scale: {cfg['scale']}×  GPU: {cfg['gpu_id']}", "info")
        pre_flags = [k for k in ("pre_autolevels","pre_jpeg_fix","pre_denoise") if cfg.get(k)]
        post_flags = [k for k in ("post_sharpen","post_contrast") if cfg.get(k)]
        tool_flags = [k for k in ("smart_crop","bg_remove") if cfg.get(k)]
        if pre_flags:  self._log(f"  Pre:  {', '.join(pre_flags)}", "info")
        if post_flags: self._log(f"  Post: {', '.join(post_flags)}", "info")
        if tool_flags: self._log(f"  Tools:{', '.join(tool_flags)}", "info")
        threading.Thread(target=self._worker, args=(snapshot, cfg), daemon=True).start()

    def cancel(self):
        if not self.running:
            return
        self._cancel.set()
        self._log("Cancellation requested…", "warn")
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════
    # HELP / ABOUT DIALOGS
    # ═══════════════════════════════════════════════════════════════════

    def _make_text_dialog(self, title: str, width: int = 80, height: int = 36) -> tuple:
        """Create a reusable scrollable text dialog. Returns (win, text_widget)."""
        c   = self.colors
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=c["bg"])
        win.resizable(True, True)
        win.geometry(f"{width * 8}x{height * 18}")
        win.transient(self.root)
        win.grab_set()

        # Close on Escape
        win.bind("<Escape>", lambda _: win.destroy())

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=8, pady=6)

        vsb = ttk.Scrollbar(frame, orient="vertical")
        hsb = ttk.Scrollbar(frame, orient="horizontal")
        txt = tk.Text(
            frame, wrap="word",
            bg=c["entry"], fg=c["fg"], insertbackground=c["fg"],
            bd=0, highlightthickness=0, font=("Segoe UI", 9),
            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
            state="normal", relief="flat",
        )
        vsb.config(command=txt.yview)
        hsb.config(command=txt.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        txt.pack(fill="both", expand=True)

        # Text tags
        txt.tag_configure("h1",   font=("Segoe UI", 12, "bold"), foreground=c["accent"],
                          spacing1=8, spacing3=4)
        txt.tag_configure("h2",   font=("Segoe UI", 10, "bold"), foreground=c["accent"],
                          spacing1=6, spacing3=2)
        txt.tag_configure("h3",   font=("Segoe UI", 9,  "bold"), foreground=c["success"],
                          spacing1=4, spacing3=1)
        txt.tag_configure("body", font=("Segoe UI", 9),          foreground=c["fg"])
        txt.tag_configure("code", font=("Consolas", 8),          foreground=c["warn"],
                          background=c["panel"])
        txt.tag_configure("tip",  font=("Segoe UI", 9, "italic"), foreground=c["border"])
        txt.tag_configure("warn", font=("Segoe UI", 9),           foreground=c["warn"])
        txt.tag_configure("ok",   font=("Segoe UI", 9),           foreground=c["success"])

        # Close button
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 6))

        return win, txt

    def _finish_text(self, txt: tk.Text):
        txt.config(state="disabled")
        txt.see("1.0")

    # ── Help window ───────────────────────────────────────────────────
    def show_help(self):
        win, txt = self._make_text_dialog("User Guide — AI Image Upscaler", 90, 42)

        def w(text, tag="body"):
            txt.insert("end", text, tag)

        w("AI Image Upscaler — User Guide\n", "h1")

        w("QUICK START\n", "h2")
        w("1. Click  + Images  or  + Folder  (or drag images onto the queue).\n")
        w("2. Choose a model in the Model tab.  Not sure? Leave it on  realesrgan-x4plus.\n")
        w("3. Click  ▶ Start.  The upscaled file saves to the Output folder.\n")
        w("4. If the result looks soft, try the  ✦ Sharpen Only  button to boost crispness.\n\n")

        w("TABS EXPLAINED\n", "h2")

        w("Model tab\n", "h3")
        w("  Model         — the NCNN upscaling model to use.  Press ℹ for a description.\n")
        w("  Scale         — 2× / 3× / 4×. Disabled for models with a fixed scale.\n")
        w("  TTA mode      — Test-Time Augmentation. 8× slower, marginally sharper edges.\n")
        w("  2-Pass        — Runs a second model on top of the first. Adds detail but doubles time.\n\n")

        w("Pre/Post tab\n", "h3")
        w("  Pre-processing  (applied BEFORE the binary runs)\n", "tip")
        w("  Auto-levels      — Normalises contrast. Good for washed-out or dark inputs.\n")
        w("  JPEG artifact    — Softens block artefacts before upscaling. Use on heavy JPEGs.\n")
        w("  NLM Denoise      — Non-Local Means denoising. h=1 light, h=10 medium, h=20 heavy.\n")
        w("                     Requires opencv (pip install opencv-python-headless).\n\n")
        w("  Post-processing  (applied AFTER the binary runs)\n", "tip")
        w("  Sharpen          — Unsharp mask. ON by default. Corrects ESRGAN's intentional softness.\n")
        w("    Presets:\n")
        w("      Photo   r=1.5 pct=130 t=3  — natural photos, avoids ringing\n", "code")
        w("      Anime   r=1.0 pct=150 t=2  — sharper edges for flat colour art\n", "code")
        w("      Art     r=2.0 pct=160 t=2  — AI-generated / illustrations\n", "code")
        w("      Strong  r=2.0 pct=200 t=1  — maximum, for very soft outputs\n", "code")
        w("      Custom               — set radius / percent / threshold manually\n", "code")
        w("\n")
        w("  ✦ Sharpen Only  — Re-sharpen already-upscaled files in the queue without\n")
        w("                    re-running the binary.  The sharpened copy saves next to the original.\n\n")
        w("  Contrast        — Multiplies contrast.  1.0 = no change.  1.05–1.10 is subtle.\n\n")

        w("Tools tab\n", "h3")
        w("  Background Removal — Uses rembg / U2Net AI. First use downloads ~170 MB model.\n")
        w("                       Saves a separate  _nobg.png  with transparent background.\n")
        w("                       Requires:  pip install \"rembg[cpu]\"\n\n")
        w("  Smart Crop       — Uses YOLOv8n to detect the main subject (person, animal, object)\n")
        w("                     and crops tightly before upscaling. Great for portraits and pets.\n")
        w("                     First use downloads ~6 MB model automatically.\n")
        w("                     Requires:  pip install ultralytics\n")
        w("  Detect classes   — Comma-separated COCO class names, e.g.  person,cat,dog,car\n")
        w("  Pad to square    — Pads the crop to equal width/height before upscaling.\n\n")
        w("  Preserve EXIF    — Copies Exif metadata (camera, GPS, date) to JPEG outputs.\n\n")

        w("Output tab\n", "h3")
        w("  Save to          — Where upscaled files are written. Click … to browse.\n")
        w("  Format           — PNG (lossless), JPEG (quality slider), WebP (quality slider).\n")
        w("  Filename         — Template for output names.  Variables:\n")
        w("    {stem}  original filename without extension\n", "code")
        w("    {model} model name used\n", "code")
        w("    {scale} scale factor (2/3/4)\n", "code")
        w("    {ext}   output extension\n", "code")
        w("    Example:  {stem}_{model}_{scale}x  →  photo_realesrgan-x4plus_4x.png\n\n", "code")

        w("Advanced tab\n", "h3")
        w("  GPU ID      — 0 = first GPU, 1 = second, -1 = CPU (very slow).\n")
        w("  Tile size   — 0 = auto. Lower values (128, 64) reduce VRAM usage.\n")
        w("                AMD RX 580 and other older GPUs: use 32 for sharpest output.\n")
        w("  Threads     — load:proc:save.  2:2:2 is a good default.\n")
        w("  Dark theme  — Toggle here or in the Advanced tab.\n\n")

        w("TIPS\n", "h2")
        w("  • For JPEG inputs: enable Pre → JPEG artifact softening for cleaner results.\n", "ok")
        w("  • For anime: use  realesrgan-x4plus-anime  and set sharpen preset to Anime.\n", "ok")
        w("  • For AI art: use  4x-UltraSharp-fp32  and sharpen preset Art.\n", "ok")
        w("  • AMD RX 580 users: prefer fp32 model variants — fp16 can cause soft output.\n", "warn")
        w("  • Very large images: set Tile size to 128 or 64 to avoid VRAM errors.\n", "warn")
        w("  • Drag and drop: drop images or folders directly onto the queue list.\n", "ok")
        w("  • Double-click or press Delete to remove items from the queue.\n", "ok")

        self._finish_text(txt)

    # ── Model Guide ───────────────────────────────────────────────────
    def show_model_guide(self):
        win, txt = self._make_text_dialog("Model Reference — AI Image Upscaler", 90, 44)

        def w(text, tag="body"):
            txt.insert("end", text, tag)

        w("Model Reference\n", "h1")
        w("All models below use the RRDB (Residual in Residual Dense Block) architecture\n")
        w("unless noted otherwise.  Only  .param + .bin  pairs in your models/ folder\n")
        w("are usable — .pth and .safetensors weights need conversion first (see below).\n\n")

        # Build a table from MODEL_DESCRIPTIONS + metadata
        sections = [
            ("GENERAL PHOTOS", [
                ("realesrgan-x4plus",       "4×", "★ Best all-rounder for real photos. 23 RRDB blocks, ~64 MB.\n"
                 "  Handles JPEG compression, blur, noise. Use this when unsure."),
                ("realesrnet-x4plus",        "4×", "Lighter than x4plus. Less texture hallucination,\n"
                 "  good when you want mild sharpening without invented detail."),
                ("realesr-general-x4v3",     "4×", "Tiny fast model (~5 MB). Low-VRAM GPUs or speed-priority."),
                ("realesr-general-wdn-x4v3", "4×", "Same tiny model with built-in wavelet denoising.\n"
                 "  Best for noisy or grainy photo inputs."),
                ("RealESRGAN_x2plus",        "2×", "Official 2× general model. Same quality as x4plus at half scale."),
            ]),
            ("ANIME / ILLUSTRATION / LINE ART", [
                ("realesrgan-x4plus-anime",    "4×", "★ Official 6-block anime model (~18 MB). Faster, smaller.\n"
                 "  Avoids sharpening smooth gradients that trip up the x4plus model."),
                ("4x-AnimeSharp",              "4×", "Sharper edges than x4plus-anime. More detail invention."),
                ("4x_NMKD-UltraYandere_300k",  "4×", "Community anime model. Strong for stylised art and manga."),
                ("4x_NMKD-YandereNeoXL_200k",  "4×", "Newer NMKD with improved edge handling on line art."),
                ("4x_BooruGan_650k",           "4×", "Trained on Danbooru art. Great for anime fan art and hair detail."),
                ("2x-AnimeSharpV3",            "2×", "2× anime upscaler. Good for moderate scale without over-processing."),
                ("2x_AniScale2_ESRGAN_i16_110K","2×","Smooth gradients, reasonable edge sharpness."),
            ]),
            ("AI-GENERATED ART / DIGITAL ILLUSTRATIONS", [
                ("4x-UltraSharp-fp32",  "4×", "★ Best for AI art on AMD GPUs. FP32 weights.\n"
                 "  Crisp edges, high contrast. Avoids fp16 accumulation issues on RX 580 etc."),
                ("4x-UltraSharp-fp16",  "4×", "Same model, FP16 weights. Preferred on NVIDIA RTX cards.\n"
                 "  Smaller memory footprint, virtually identical quality."),
                ("4x-UltraSharp",       "4×", "Original UltraSharp PTH weights (not NCNN-ready until converted)."),
                ("4x-UltraSharpV2",     "4×", "V2 — better texture fidelity, less ringing. Needs conversion."),
            ]),
            ("VIDEO FRAMES", [
                ("realesr-animevideov3-x2", "2×", "★ Anime video at 2×. Tuned for temporal consistency — low flicker."),
                ("realesr-animevideov3-x3", "3×", "Same temporal tuning, 3× scale."),
                ("realesr-animevideov3-x4", "4×", "Most common animevideov3 variant for HD upscaling."),
            ]),
            ("RESTORATION WITHOUT UPSCALING", [
                ("1x-UnResizeOnly_RCAN", "1×", "RCAN architecture. Cleans JPEG artefacts without changing resolution.\n"
                 "  Use as a pre-pass before another model."),
            ]),
        ]

        for section_title, models in sections:
            w(f"\n{section_title}\n", "h2")
            for name, scale, desc in models:
                # Check if NCNN pair exists
                param_ok = (MODELS_DIR / f"{name}.param").exists()
                bin_ok   = (MODELS_DIR / f"{name}.bin").exists()
                if param_ok and bin_ok:
                    status = "✓ NCNN ready"
                    stag   = "ok"
                elif (MODELS_DIR / f"{name}.pth").exists() or (MODELS_DIR / f"{name}.safetensors").exists():
                    status = "⚠ PTH only — needs conversion"
                    stag   = "warn"
                else:
                    status = "✗ not in models/"
                    stag   = "warn"

                w(f"  {name}", "h3")
                w(f"  ({scale})  ")
                w(f"[{status}]\n", stag)
                for line in desc.split("\n"):
                    w(f"    {line}\n", "body")

        w("\nCONVERTING PTH → NCNN\n", "h2")
        w("  If a model shows '⚠ PTH only', run the included converter:\n\n")
        w('  python "model converter.py"\n', "code")
        w("\n  Requirements: torch, realesrgan, safetensors, onnx, pnnx or onnx2ncnn\n")
        w("  See requirements.txt for full instructions.\n\n")

        w("ADDING NEW MODELS\n", "h2")
        w("  1. Download a .param + .bin pair from OpenModelDB (openmodeldb.info)\n")
        w("     or convert a .pth file using the model converter.\n")
        w("  2. Drop both files into the  models/  folder next to the app.\n")
        w("  3. Click  Tools → Refresh Model List  in the app.\n")

        self._finish_text(txt)

    # ── About dialog ──────────────────────────────────────────────────
    def show_about(self):
        c   = self.colors
        win = tk.Toplevel(self.root)
        win.title("About — AI Image Upscaler")
        win.configure(bg=c["bg"])
        win.resizable(False, False)
        win.geometry("480x520")
        win.transient(self.root)
        win.grab_set()
        win.bind("<Escape>", lambda _: win.destroy())

        # Try to show the icon as a large image
        try:
            icon_src = APP_DIR / "assets" / "icon_512.png"
            if not icon_src.exists():
                icon_src = APP_DIR / "icon.ico"
            if icon_src.exists() and PIL_AVAILABLE:
                with Image.open(str(icon_src)) as im:
                    im.thumbnail((80, 80), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(im)
                lbl_ico = tk.Label(win, image=photo, bg=c["bg"])
                lbl_ico.image = photo
                lbl_ico.pack(pady=(18, 4))
        except Exception:
            pass

        tk.Label(win, text="AI Image Upscaler",
                 font=("Segoe UI", 16, "bold"),
                 fg=c["accent"], bg=c["bg"]).pack()
        tk.Label(win, text="Version 1.0.1",
                 font=("Segoe UI", 9),
                 fg=c["border"], bg=c["bg"]).pack()
        tk.Label(win, text="A desktop GUI for Real-ESRGAN ncnn-Vulkan",
                 font=("Segoe UI", 9),
                 fg=c["fg"], bg=c["bg"]).pack(pady=(4, 0))
        tk.Label(win, text="Runs fully offline  ·  AMD, NVIDIA, Intel GPU via Vulkan",
                 font=("Segoe UI", 9),
                 fg=c["border"], bg=c["bg"]).pack()

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=24, pady=12)

        credits_frame = ttk.Frame(win)
        credits_frame.pack(padx=24)

        credits = [
            ("Upscaling engine", "Real-ESRGAN ncnn-Vulkan",
             "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan"),
            ("AI model research",  "Real-ESRGAN (Xintao Wang et al.)",
             "https://github.com/xinntao/Real-ESRGAN"),
            ("UltraSharp models",  "Kim2091 (Hugging Face)",
             "https://huggingface.co/Kim2091/UltraSharp"),
            ("Background removal", "rembg (Daniel Gatis)",
             "https://github.com/danielgatis/rembg"),
            ("Object detection",   "Ultralytics YOLOv8",
             "https://github.com/ultralytics/ultralytics"),
            ("Drag-and-drop",      "tkinterdnd2",
             "https://github.com/pmgagne/tkinterdnd2"),
            ("Model repository",   "OpenModelDB",
             "https://openmodeldb.info"),
        ]

        for label, name, url in credits:
            row = ttk.Frame(credits_frame)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:", width=20, anchor="e",
                     font=("Segoe UI", 8), fg=c["border"], bg=c["bg"]).pack(side="left")
            btn = tk.Label(row, text=name, anchor="w",
                           font=("Segoe UI", 8, "underline"),
                           fg=c["accent"], bg=c["bg"], cursor="hand2")
            btn.pack(side="left", padx=6)
            btn.bind("<Button-1>", lambda e, u=url: webbrowser.open(u))

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=24, pady=12)

        tk.Label(win, text="MIT License  ·  https://github.com/Zushikina-kun/AIU-Image-AI-Upscaler",
                 font=("Segoe UI", 8),
                 fg=c["border"], bg=c["bg"], cursor="hand2").pack()
        tk.Label(win, text="",
                 bg=c["bg"]).pack(pady=2)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=8)

    # ═══════════════════════════════════════════════════════════════════
    # SETTINGS PERSISTENCE
    # ═══════════════════════════════════════════════════════════════════
    def _on_close(self):
        save_settings({
            "model":          self.var_model.get(),
            "scale":          self.var_scale.get(),
            "tile":           self.var_tile.get(),
            "threads":        self.var_threads.get(),
            "output_dir":     self.var_outdir.get(),
            "out_format":     self.var_format.get(),
            "jpeg_quality":   self.var_jpeg_q.get(),
            "webp_quality":   self.var_webp_q.get(),
            "phase2":         self.var_phase2.get(),
            "model2":         self.var_model2.get(),
            "dark_theme":     self.var_dark.get(),
            "gpu_id":         self.var_gpu.get(),
            "tta_mode":       self.var_tta.get(),
            "pre_denoise":    self.var_pre_denoise.get(),
            "denoise_h":      self.var_denoise_h.get(),
            "pre_autolevels": self.var_pre_autolevels.get(),
            "pre_jpeg_fix":   self.var_pre_jpeg_fix.get(),
            "post_sharpen":   self.var_post_sharpen.get(),
            "sharpen_preset": self.var_sharpen_preset.get(),
            "sharpen_radius": self.var_sharpen_radius.get(),
            "sharpen_pct":    self.var_sharpen_pct.get(),
            "sharpen_thresh": self.var_sharpen_thresh.get(),
            "post_contrast":  self.var_post_contrast.get(),
            "contrast_factor":self.var_contrast_f.get(),
            "bg_remove":      self.var_bg_remove.get(),
            "smart_crop":     self.var_smart_crop.get(),
            "smart_pad":      self.var_smart_pad.get(),
            "detect_classes": self.var_detect_classes.get(),
            "preserve_exif":  self.var_preserve_exif.get(),
            "name_template":  self.var_name_template.get(),
        })
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    root.geometry("1150x740")
    UpscalerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
