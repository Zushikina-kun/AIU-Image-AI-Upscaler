# -*- mode: python ; coding: utf-8 -*-
#
# Build:  pyinstaller main.spec
# Requires: pip install pyinstaller pillow tkinterdnd2 opencv-python-headless
#           rembg[cpu] ultralytics piexif

import sys
from pathlib import Path

APP_DIR = Path(SPECPATH)

# ── tkinterdnd2 ────────────────────────────────────────────────────────────────
# The package ships platform-specific DLL folders. We need the full tkdnd
# directory tree so Tcl/Tk can find the pkgIndex.tcl at runtime.
try:
    import tkinterdnd2
    dnd_pkg_dir = Path(tkinterdnd2.__file__).parent
    dnd_tkdnd   = dnd_pkg_dir / "tkdnd"
    dnd_bins    = []
    dnd_datas   = [
        (str(dnd_pkg_dir / "TkinterDnD.py"), "tkinterdnd2"),
        (str(dnd_pkg_dir / "__init__.py"),    "tkinterdnd2"),
        (str(dnd_tkdnd),                      "tkinterdnd2/tkdnd"),
    ]
except ImportError:
    dnd_bins  = []
    dnd_datas = []

# ── onnxruntime providers (rembg needs these) ─────────────────────────────────
try:
    import onnxruntime
    ort_dir   = Path(onnxruntime.__file__).parent
    ort_libs  = [(str(p), "onnxruntime/capi")
                 for p in (ort_dir / "capi").glob("*.dll")]
except Exception:
    ort_libs  = []

# ── rembg u2net model path hint ───────────────────────────────────────────────
# rembg downloads models to ~/.u2net/ at runtime — nothing to bundle.

a = Analysis(
    ["main.py"],
    pathex=[str(APP_DIR)],
    binaries=ort_libs,
    datas=dnd_datas + [
        (str(APP_DIR / "icon.ico"),  "."),
        # Bundle the inference binary and required DLLs
        (str(APP_DIR / "realesrgan-ncnn-vulkan.exe"), "."),
        (str(APP_DIR / "vcomp140.dll"),               "."),
        (str(APP_DIR / "vcomp140d.dll"),              "."),
    ],
    hiddenimports=[
        # Pillow
        "PIL._tkinter_finder",
        "PIL.Image", "PIL.ImageTk", "PIL.ImageFilter",
        "PIL.ImageOps", "PIL.ImageEnhance",
        # DnD
        "tkinterdnd2", "tkinterdnd2.TkinterDnD",
        # OpenCV
        "cv2",
        # rembg + onnxruntime
        "rembg",
        "onnxruntime", "onnxruntime.capi", "onnxruntime.capi.onnxruntime_pybind11_state",
        # numpy (needed by cv2 and rembg)
        "numpy", "numpy.core._multiarray_umath",
        # piexif
        "piexif",
        # ultralytics (lazy-loaded but PyInstaller needs the hint)
        "ultralytics",
        # stdlib that PyInstaller sometimes misses
        "pathlib", "json", "queue", "threading", "webbrowser",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude heavy training-only packages — not needed at runtime
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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AI Image Upscaler",
    icon=str(APP_DIR / "icon.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        # Don't UPX-compress DLLs that break when compressed
        "vcruntime*.dll", "vcomp*.dll", "python*.dll",
        "onnxruntime*.dll", "_onnxruntime*.pyd",
    ],
    runtime_tmpdir=None,
    console=False,              # no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
