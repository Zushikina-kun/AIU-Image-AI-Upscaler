"""
Post-build script: copies models/ and the inference binary into
dist/AI Image Upscaler/ so the app works out of the box.
Run after: pyinstaller main.spec --noconfirm
"""
import os, shutil, sys
from pathlib import Path

proj = Path(__file__).parent
dist = proj / "dist" / "AI Image Upscaler"

if not dist.exists():
    print(f"ERROR: dist folder not found: {dist}")
    print("Run 'pyinstaller main.spec --noconfirm' first.")
    sys.exit(1)

# ── Copy inference binary + DLLs ─────────────────────────────────────────────
for fname in ("realesrgan-ncnn-vulkan.exe", "vcomp140.dll", "vcomp140d.dll"):
    src = proj / fname
    dst = dist / fname
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied {fname}  ({src.stat().st_size // 1024} KB)")
    else:
        print(f"  WARN: {fname} not found in project root")

# ── Copy models/ ──────────────────────────────────────────────────────────────
models_src = proj / "models"
models_dst = dist / "models"
models_dst.mkdir(exist_ok=True)

copied = 0
for f in models_src.iterdir():
    if f.is_file():
        shutil.copy2(f, models_dst / f.name)
        copied += 1

params = len(list(models_dst.glob("*.param")))
bins   = len(list(models_dst.glob("*.bin")))
print(f"  Copied models/: {copied} files  ({params} .param  {bins} .bin)")

# ── Summary ───────────────────────────────────────────────────────────────────
total = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
exe   = dist / "AI Image Upscaler.exe"
print()
print(f"Dist folder   : {dist}")
print(f"EXE size      : {exe.stat().st_size // 1024 // 1024} MB")
print(f"Total size    : {total // 1024 // 1024} MB")
print(f"Binary OK     : {(dist / 'realesrgan-ncnn-vulkan.exe').exists()}")
print(f"Models .param : {params}")
print(f"Models .bin   : {bins}")
print()
print("Done. You can now run:")
print(f'  "{dist}\\AI Image Upscaler.exe"')
