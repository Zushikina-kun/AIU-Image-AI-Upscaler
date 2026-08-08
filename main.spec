# -*- mode: python ; coding: utf-8 -*-
#
# Build:  pyinstaller main.spec
# Requires:  pip install pyinstaller pillow tkinterdnd2

import sys
from pathlib import Path

APP_DIR = Path(SPECPATH)

# Collect tkinterdnd2 binaries (the platform DLL that enables drag-and-drop)
try:
    import tkinterdnd2
    dnd_dir  = Path(tkinterdnd2.__file__).parent
    dnd_bins = [(str(p), "tkinterdnd2") for p in dnd_dir.rglob("*.dll")]
    dnd_bins += [(str(p), "tkinterdnd2") for p in dnd_dir.rglob("*.so")]
    dnd_data = [(str(dnd_dir / "tkdnd"), "tkinterdnd2/tkdnd")]
except ImportError:
    dnd_bins = []
    dnd_data = []

a = Analysis(
    ["main.py"],
    pathex=[str(APP_DIR)],
    binaries=dnd_bins,
    datas=dnd_data + [
        # Bundle the icon
        ("icon.ico", "."),
    ],
    hiddenimports=[
        "PIL._tkinter_finder",
        "tkinterdnd2",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "basicsr", "realesrgan"],  # not needed at runtime
    noarchive=False,
    optimize=0,
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
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
