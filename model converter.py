"""
model converter.py
──────────────────
Converts Real-ESRGAN / ESRGAN .pth (or .safetensors) model weights to the
NCNN .param/.bin format consumed by realesrgan-ncnn-vulkan.exe.

Pipeline
────────
  1. Load the PyTorch model from the .pth/.safetensors file
  2. Export to ONNX  (torch.onnx.export)
  3. Convert ONNX → NCNN  (requires onnx2ncnn.exe on PATH, ships with ncnn)

Requirements (install via requirements.txt):
  pip install torch torchvision realesrgan safetensors onnx

onnx2ncnn must be on your PATH or set ONNX2NCNN_EXE below.
It ships inside the ncnn release archive:
  https://github.com/Tencent/ncnn/releases

Usage
─────
  python "model converter.py"                     # converts all .pth/.safetensors
  python "model converter.py" --dry-run           # show what would be done
  python "model converter.py" --model 4x-UltraSharp.pth
  python "model converter.py" --fp16              # export FP16 weights (smaller)
  python "model converter.py" --scale 4           # override scale (default: auto-detect)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ── Optional heavy imports (only loaded when actually converting) ─────────────
def _require(pkg: str):
    try:
        return __import__(pkg)
    except ImportError:
        print(f"[!] Missing package: '{pkg}'. Run:  pip install {pkg}")
        sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR      = Path(__file__).parent / "models"
TILE           = 0
TILE_PAD       = 10
PRE_PAD        = 0

# Path to onnx2ncnn tool — set to full path if not on PATH
ONNX2NCNN_EXE  = os.environ.get("ONNX2NCNN_EXE", "onnx2ncnn")

# ── Architecture registry ─────────────────────────────────────────────────────
# Maps a lowercase keyword in the model filename → (num_feat, num_block, scale)
# Extend this table when you add new model families.
ARCH_REGISTRY: list[tuple[str, int, int, int]] = [
    # keyword,               num_feat, num_block, scale
    ("realesrgan-x4plus-anime",  64,       6,     4),   # 6-block lite anime model
    ("realesr-animevideov3",     64,       6,     4),   # animevideov3 variants
    ("realesrgan",               64,      23,     4),   # standard x4plus
    ("realesrnet",               64,      23,     4),
    ("4x-ultrasharp",            64,      23,     4),
    ("4x-animesharp",            64,      23,     4),
    ("4x_nmkd",                  64,      23,     4),
    ("4x_boorugan",              64,      23,     4),
    ("2x-sudo",                  64,      23,     2),
    ("2x-aniscale",              64,      23,     2),
    ("2x-animesharp",            64,      23,     2),
    ("1x-unresize",              64,      23,     1),
]


def detect_arch(name: str) -> tuple[int, int, int]:
    """Return (num_feat, num_block, scale) for a given model filename stem."""
    lower = name.lower()
    for keyword, nf, nb, sc in ARCH_REGISTRY:
        if keyword in lower:
            return nf, nb, sc
    # Fallback: try to read scale from name like "2x_..." or "4x_..."
    for prefix, sc in (("4x", 4), ("3x", 3), ("2x", 2), ("1x", 1)):
        if lower.startswith(prefix) or f"_{prefix}" in lower or f"-{prefix}" in lower:
            return 64, 23, sc
    return 64, 23, 4   # safe default


def load_model_weights(pth_path: Path, arch, scale: int, fp16: bool):
    """Load .pth or .safetensors weights into the architecture and return the model."""
    torch = _require("torch")

    suffix = pth_path.suffix.lower()
    if suffix == ".safetensors":
        safetensors_torch = _require("safetensors.torch")
        state_dict = safetensors_torch.load_file(str(pth_path))
    else:
        state_dict = torch.load(str(pth_path), map_location="cpu")
        # Some checkpoints wrap weights under a 'params_ema' or 'params' key
        for key in ("params_ema", "params"):
            if key in state_dict:
                state_dict = state_dict[key]
                break

    arch.load_state_dict(state_dict, strict=True)
    arch.eval()
    if fp16:
        arch = arch.half()
    return arch


def export_onnx(model, scale: int, tmp_dir: Path, name: str, fp16: bool) -> Path:
    """Export model to ONNX and return the output path."""
    torch = _require("torch")

    onnx_path = tmp_dir / f"{name}.onnx"
    dtype     = torch.float16 if fp16 else torch.float32

    # Dummy input: 1 × 3 × 64 × 64 (tile size used by ncnn binary)
    dummy = torch.zeros(1, 3, 64, 64, dtype=dtype)

    print(f"    Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input":  {2: "h", 3: "w"},
                      "output": {2: "H", 3: "W"}},
        do_constant_folding=True,
    )
    return onnx_path


def convert_onnx_to_ncnn(onnx_path: Path, param_out: Path, bin_out: Path) -> bool:
    """Call onnx2ncnn to produce .param/.bin.  Returns True on success."""
    exe = shutil.which(ONNX2NCNN_EXE) or ONNX2NCNN_EXE
    cmd = [exe, str(onnx_path), str(param_out), str(bin_out)]
    print(f"    Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"    {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"    [!] onnx2ncnn error:\n{result.stderr.strip()}")
            return False
        return True
    except FileNotFoundError:
        print(f"[!] onnx2ncnn not found at '{exe}'.")
        print("    Download it from https://github.com/Tencent/ncnn/releases")
        print("    and either put it on your PATH or set the ONNX2NCNN_EXE env var.")
        return False


def convert_model(pth_path: Path, scale_override: int | None,
                  fp16: bool, dry_run: bool) -> bool:
    """Full pipeline for one model file. Returns True on success."""
    torch    = _require("torch")
    RRDBNet  = _require("basicsr.archs.rrdbnet_arch").RRDBNet

    name       = pth_path.stem
    param_out  = MODEL_DIR / f"{name}.param"
    bin_out    = MODEL_DIR / f"{name}.bin"

    if param_out.exists() and bin_out.exists():
        print(f"[✓] Already converted, skipping: {name}")
        return True

    num_feat, num_block, auto_scale = detect_arch(name)
    scale = scale_override if scale_override is not None else auto_scale

    print(f"\n[•] {pth_path.name}")
    print(f"    arch: RRDBNet  feat={num_feat}  blocks={num_block}  scale={scale}"
          f"{'  fp16' if fp16 else ''}")

    if dry_run:
        print("    [dry-run] would export to ONNX then convert to NCNN")
        return True

    # Build architecture
    arch = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=num_feat, num_block=num_block,
        num_grow_ch=32, scale=scale,
    )

    # Load weights
    print(f"    Loading weights…")
    try:
        arch = load_model_weights(pth_path, arch, scale, fp16)
    except Exception as e:
        print(f"    [!] Weight load failed: {e}")
        print("    Hint: the model may use a different architecture.")
        print("    Edit ARCH_REGISTRY in this script to add it.")
        return False

    # Export + convert inside a temp dir; move on success
    with tempfile.TemporaryDirectory(prefix="ncnn_conv_") as tmp:
        tmp_dir = Path(tmp)
        try:
            onnx_path = export_onnx(arch, scale, tmp_dir, name, fp16)
        except Exception as e:
            print(f"    [!] ONNX export failed: {e}")
            return False

        tmp_param = tmp_dir / f"{name}.param"
        tmp_bin   = tmp_dir / f"{name}.bin"
        ok = convert_onnx_to_ncnn(onnx_path, tmp_param, tmp_bin)

        if ok and tmp_param.exists() and tmp_bin.exists():
            shutil.copy2(tmp_param, param_out)
            shutil.copy2(tmp_bin,   bin_out)
            print(f"[✓] Saved: {param_out.name}, {bin_out.name}")
            return True
        else:
            print(f"[!] Conversion produced no output files.")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert Real-ESRGAN .pth/.safetensors → NCNN .param/.bin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",   metavar="FILE",
                        help="Convert only this specific file (relative to models/)")
    parser.add_argument("--scale",   type=int, choices=[1, 2, 3, 4],
                        help="Override scale factor (default: auto-detected from filename)")
    parser.add_argument("--fp16",    action="store_true",
                        help="Export FP16 weights (smaller .bin, minor quality loss)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be converted without doing it")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        print(f"[!] models/ directory not found at {MODEL_DIR}")
        sys.exit(1)

    # Gather target files
    exts = {".pth", ".safetensors"}
    if args.model:
        targets = [MODEL_DIR / args.model]
    else:
        targets = sorted(p for p in MODEL_DIR.iterdir() if p.suffix.lower() in exts)

    if not targets:
        print("[!] No .pth or .safetensors files found in models/")
        sys.exit(0)

    print(f"Found {len(targets)} model(s) to process.\n")
    succeeded, failed, skipped = 0, 0, 0

    for pth_path in targets:
        if not pth_path.exists():
            print(f"[!] File not found: {pth_path}")
            failed += 1
            continue
        name      = pth_path.stem
        param_out = MODEL_DIR / f"{name}.param"
        bin_out   = MODEL_DIR / f"{name}.bin"
        if param_out.exists() and bin_out.exists():
            print(f"[✓] Already converted, skipping: {name}")
            skipped += 1
            continue
        ok = convert_model(pth_path, args.scale, args.fp16, args.dry_run)
        if ok:
            succeeded += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Done.  Converted: {succeeded}  Skipped: {skipped}  Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
