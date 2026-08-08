# -*- mode: python ; coding: utf-8 -*-
#
# Build:  pyinstaller main.spec --noconfirm
# Requires (in venv_build):
#   pip install pyinstaller pillow tkinterdnd2 opencv-python-headless
#              rembg[cpu] ultralytics piexif
#
# Output: dist\AI Image Upscaler\AI Image Upscaler.exe
#         (onedir mode so models/ and the binary live next to the EXE on disk)

import sys
from pathlib import Path

APP_DIR = Path(SPECPATH)

# ── tkinterdnd2 ───────────────────────────────────────────────────────────────
try:
    import tkinterdnd2
    dnd_pkg_dir = Path(tkinterdnd2.__file__).parent
    dnd_tkdnd   = dnd_pkg_dir / "tkdnd"
    dnd_datas   = [
        (str(dnd_pkg_dir / "TkinterDnD.py"), "tkinterdnd2"),
        (str(dnd_pkg_dir / "__init__.py"),    "tkinterdnd2"),
        (str(dnd_tkdnd),                      "tkinterdnd2/tkdnd"),
    ]
except ImportError:
    dnd_datas = []

# ── onnxruntime provider DLLs (rembg) ────────────────────────────────────────
try:
    import onnxruntime
    ort_dir  = Path(onnxruntime.__file__).parent
    ort_libs = [(str(p), "onnxruntime/capi")
                for p in (ort_dir / "capi").glob("*.dll")]
except Exception:
    ort_libs = []

a = Analysis(
    ["main.py"],
    pathex=[str(APP_DIR)],
    binaries=ort_libs,
    datas=dnd_datas + [
        # App icon only — NOT the ncnn binary or models/
        # Those are copied post-build alongside the EXE so the user can add models easily
        (str(APP_DIR / "icon.ico"), "."),
    ],
    hiddenimports=[
        "PIL._tkinter_finder",
        "PIL.Image", "PIL.ImageTk", "PIL.ImageFilter",
        "PIL.ImageOps", "PIL.ImageEnhance",
        "tkinterdnd2", "tkinterdnd2.TkinterDnD",
        "cv2",
        "rembg",
        "onnxruntime", "onnxruntime.capi",
        "onnxruntime.capi.onnxruntime_pybind11_state",
        "numpy", "numpy.core._multiarray_umath",
        "piexif",
        "ultralytics",
        "pathlib", "json", "queue", "threading", "webbrowser",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "torchaudio",
        "basicsr", "realesrgan",
        "matplotlib", "IPython", "jupyter",
        "scipy", "sklearn", "pandas",
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

# ── EXE (the launcher, no embedded data — everything stays on disk) ───────────
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,          # binaries go into COLLECT, not EXE
    name="AI Image Upscaler",
    icon=str(APP_DIR / "icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=["vcruntime*.dll", "vcomp*.dll", "python*.dll",
                 "onnxruntime*.dll", "_onnxruntime*.pyd"],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# ── COLLECT — creates dist\AI Image Upscaler\ folder ─────────────────────────
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=["vcruntime*.dll", "vcomp*.dll", "python*.dll",
                 "onnxruntime*.dll", "_onnxruntime*.pyd"],
    name="AI Image Upscaler",
)
