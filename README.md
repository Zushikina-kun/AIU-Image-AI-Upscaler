# AI Image Upscaler

A Windows desktop app for AI-powered image upscaling using [Real-ESRGAN ncnn-Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan).  
Runs fully **offline** — no cloud, no subscriptions, no CUDA. AMD, NVIDIA, and Intel GPUs all work via Vulkan.

![screenshot](assets/screenshot.png)

---

## Download

**[→ Latest release: v1.2.0](https://github.com/Zushikina-kun/AIU-Image-AI-Upscaler/releases/latest)**

Download `AI.Image.Upscaler.v1.2.0.zip`, extract it, and run `AI Image Upscaler.exe`.  
No Python, no installation, no CUDA required.

---

## Features

| Category | What it does |
|---|---|
| **Upscaling** | 2×, 3×, 4× via any NCNN `.param/.bin` model — 8 built-in, unlimited custom |
| **2-Pass enhance** | Chain two models (e.g. x4plus → UltraSharp) for maximum detail |
| **Sharpening presets** | Photo / Anime / Art / Strong / Custom — corrects ESRGAN's intentional softness |
| **Sharpen-Only mode** | Re-sharpen already-upscaled files without re-running the binary |
| **Pre-processing** | Auto-levels, JPEG artifact softening, NLM denoising (strength slider) |
| **Post-processing** | Unsharp mask + contrast enhancement |
| **Background removal** | via [rembg](https://github.com/danielgatis/rembg) / U2Net — saves `_nobg.png` |
| **Smart crop** | Auto-detects persons/animals/objects via YOLOv8n, crops to subject before upscaling |
| **Output formats** | PNG (lossless), JPEG (quality slider), WebP (quality slider) |
| **Filename templates** | `{stem}_{model}_{scale}x` — fully configurable |
| **EXIF preservation** | Copies EXIF metadata from source to JPEG output |
| **Batch queue** | Drag-and-drop, add folders (recursive), reorder, multi-select remove, ETA |
| **Before/After preview** | Side-by-side canvases, resize with window, resolution + file size labels |
| **Live log** | Colour-coded subprocess output per image |
| **Built-in Help** | User Guide, Model Reference, About dialog — all in the Help menu |
| **Settings persistence** | Everything saved to `settings.json` on close |
| **Dark / Light theme** | Toggle in the Advanced tab |
| **Cancel** | Kills the running subprocess immediately |

---

## Quick Start (Portable EXE)

```
1. Download AI.Image.Upscaler.v1.2.0.zip from the Releases page
2. Extract the ZIP — keep all files together in one folder
3. Double-click  AI Image Upscaler.exe
4. Click + Images or drag images onto the queue
5. Choose a model, click ▶ Start
```

> The `models/`, `realesrgan-ncnn-vulkan.exe`, and `_internal/` folders must stay  
> in the same directory as the EXE. Do not move the EXE out of its folder.

---

## Quick Start (From Source)

```powershell
git clone https://github.com/Zushikina-kun/AIU-Image-AI-Upscaler.git
cd "AIU-Image-AI-Upscaler"
python -m pip install -r requirements.txt
python main.py
```

> `rembg` downloads ~170 MB on first background-removal use.  
> `ultralytics` downloads ~6 MB on first smart-crop use.

---

## NCNN Models Included

All models below have working `.param` + `.bin` pairs and are tested on AMD RX 580 + NVIDIA GPUs.

| Model | Scale | Best for | Notes |
|---|---|---|---|
| `realesrgan-x4plus` | 4× | **General photos** — the safe default | 23 RRDB blocks, ~64 MB weights |
| `realesrgan-x4plus-anime` | 4× | **Anime / flat-colour art** | 6 blocks, faster and smaller |
| `realesrnet-x4plus` | 4× | Lighter general upscaling | Less texture hallucination than x4plus |
| `realesr-animevideov3-x2` | 2× | **Video frames** | Temporal consistency, low flicker |
| `realesr-animevideov3-x3` | 3× | Video frames | |
| `realesr-animevideov3-x4` | 4× | Video frames | Most common variant |
| `4x-UltraSharp-fp16` | 4× | **AI-generated art** — crisp edges | Use on NVIDIA; fp16 throughput |
| `4x-UltraSharp-fp32` | 4× | **AI-generated art on AMD** | fp32 avoids RX 580 fp16 precision bug |

### Choosing the right model

| Your image | Best model |
|---|---|
| Real-world photo | `realesrgan-x4plus` |
| Anime / manga / line art | `realesrgan-x4plus-anime` |
| AI-generated art (NVIDIA) | `4x-UltraSharp-fp16` |
| AI-generated art (AMD RX 580+) | `4x-UltraSharp-fp32` |
| Screenshot / digital art | `4x-UltraSharp-fp32` |
| Video frames | `realesr-animevideov3-x4` |

### Adding more models

1. Get a `.param` + `.bin` pair from [OpenModelDB](https://openmodeldb.info)
2. Drop both files into the `models/` folder next to the EXE
3. Click **Tools → Refresh Model List** in the app

---

## Sharpening Guide

ESRGAN is trained to avoid over-sharpening — output is intentionally slightly soft to prevent ringing artifacts. Post-sharpening restores crispness. The app has **four ready-to-use presets**:

| Preset | Settings | Best for |
|---|---|---|
| **Photo** | r=1.5, 130%, t=3 | Natural photos — moderate sharpening, no ringing |
| **Anime** | r=1.0, 150%, t=2 | Anime/line art — sharper edges on flat colour |
| **Art** | r=2.0, 160%, t=2 | AI-generated illustrations — strong detail |
| **Strong** | r=2.0, 200%, t=1 | Very soft outputs — maximum sharpening |
| **Custom** | manual | Set radius, percent, threshold yourself |

**Sharpen-Only button:** Add already-upscaled images to the queue, click this to re-sharpen them without re-running the binary. Saves as `_sharp` copies.

**Measured result:** UnsharpMask (Photo preset) adds +14% edge clarity on real upscale outputs.

---

## Pre & Post Processing

### Pre-processing (applied before upscaling)

| Option | What it does | When to use |
|---|---|---|
| **Auto-levels** | `ImageOps.autocontrast` — normalises exposure | Washed-out or over-exposed inputs |
| **JPEG artifact softening** | Median filter reduces block artefacts | Heavily compressed JPEGs |
| **NLM Denoise** h= | OpenCV Non-Local Means — h=1 light, 10 medium, 20+ heavy | Grainy/noisy photos |

### Post-processing (applied after upscaling)

| Option | Default | Notes |
|---|---|---|
| **Sharpen** | **On** (Photo preset) | Unsharp mask — see presets above |
| **Contrast** | Off | `ImageEnhance.Contrast` — 1.0 = no change |

---

## Tools

### Background Removal
Uses [rembg](https://github.com/danielgatis/rembg) / U2Net. Enable in the **Tools** tab.  
Saves a `_nobg.png` alongside the upscaled output. First use downloads ~170 MB.

```
pip install "rembg[cpu]"
```

### Smart Crop
Uses YOLOv8n. Detects the largest subject (person, animal, etc.) and crops to it before upscaling.  
First use downloads ~6 MB.

```
pip install ultralytics
```

- **Detect classes:** comma-separated COCO labels, e.g. `person,cat,dog,car`  
- **Pad to square:** ensures equal width/height before upscaling (recommended)

---

## Output Settings

### Filename Template

Default: `{stem}_{model}_{scale}x`

| Variable | Replaced with |
|---|---|
| `{stem}` | Original filename without extension |
| `{model}` | Model name used |
| `{scale}` | Scale factor (2, 3, or 4) |
| `{ext}` | Output file extension |

Example: `cat_realesrgan-x4plus_4x.png`

---

## Advanced Settings

| Setting | Default | Notes |
|---|---|---|
| **GPU ID** | `0` | `0` = first GPU. Set `0,1` for multi-GPU. |
| **Tile size** | `32` | `0` = auto. Use `32` on AMD Polaris/Vega for sharpest output. Lower for less VRAM. |
| **Threads** | `2:2:2` | `load:proc:save` — increase for batches of small images. |
| **TTA mode** | Off | Test-Time Augmentation — 8× slower, marginally sharper. |

> **Why tile=32 by default?**  
> On AMD Polaris/Vega GPUs (e.g. RX 580), fp16 accumulation is flagged as broken.  
> Smaller tiles reduce the numerical error per pass, producing measurably sharper output.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| **"Executable not found"** | `realesrgan-ncnn-vulkan.exe` must be in the **same folder** as the EXE. Don't move the EXE out of the extracted ZIP folder. |
| **No models in dropdown** | `models/` folder must be in the same folder as the EXE. Check it has `.param` + `.bin` files. |
| **Output is soft / blurry** | Sharpening is on by default. If still soft, try **Strong** preset or increase percent to 180%. For JPEG inputs, also enable Pre → JPEG artifact softening. |
| **AMD GPU — soft output** | Use `4x-UltraSharp-fp32` (not fp16). The RX 580/Polaris has a known fp16 precision bug in Vulkan. |
| **Black output image** | The `realesr-general-x4v3` model is currently excluded — its NCNN conversion produces black output. Use `realesrgan-x4plus` instead. |
| **VRAM out of memory** | Lower Tile size to `128`, `64`, or `32` in Advanced tab. |
| **Background removal fails** | Run `pip install "rembg[cpu]"` and restart. |
| **Smart crop fails** | Run `pip install ultralytics` and restart. |
| **Model not in list** | Drop `.param` + `.bin` files into `models/`, then click **Tools → Refresh Model List**. |
| **Slow first run** | rembg (~170 MB) and YOLOv8n (~6 MB) download automatically on first use. |

---

## Building from Source

```powershell
# 1. Create a clean build venv
python -m venv venv_build
.\venv_build\Scripts\pip install Pillow tkinterdnd2 opencv-python-headless "rembg[cpu]" ultralytics piexif pyinstaller

# 2. Build
.\venv_build\Scripts\python -m PyInstaller main.spec --noconfirm

# 3. Copy inference binary + models
python post_build.py

# Output: dist\AI Image Upscaler\AI Image Upscaler.exe
```

> Use a **clean venv** with only the listed packages. The system Python may have torch/scipy/sympy  
> installed which bloat the EXE from ~60 MB to 200+ MB.

---

## Video Upscaling (Manual Workflow)

```powershell
# 1. Extract frames
mkdir tmp_frames
ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 tmp_frames/frame%08d.jpg

# 2. Add tmp_frames/ to the queue, select realesr-animevideov3-x4, click Start

# 3. Merge back with audio
ffmpeg -i out_frames/frame%08d.png -i video.mp4 -map 0:v:0 -map 1:a:0 `
       -c:a copy -c:v libx264 -r 23.98 -pix_fmt yuv420p output_upscaled.mp4
```

---

## Project Structure

```
AIU-Image-AI-Upscaler/
├── main.py                    # Main application (~1,900 lines)
├── model converter.py         # PTH/safetensors → NCNN conversion utility
├── post_build.py              # Post-PyInstaller copy script
├── main.spec                  # PyInstaller build spec (onedir)
├── requirements.txt           # Python runtime dependencies
├── icon.ico                   # App icon (7 sizes: 16–256px)
├── assets/
│   ├── screenshot.png         # App preview
│   └── icon_512.png           # 512px icon for GitHub
├── realesrgan-ncnn-vulkan.exe # Vulkan inference binary
├── vcomp140.dll               # Visual C++ runtime
├── vcomp140d.dll              # Visual C++ runtime (debug)
├── onepiece_demo.mp4          # Demo video for frame workflow
└── models/
    ├── *.param                # NCNN model graph files  (8 pairs)
    ├── *.bin                  # NCNN model weights
    ├── *.pth                  # PyTorch weights (not used at runtime)
    └── requirements_pth2ncnn.txt
```

---

## Credits

| Component | Author |
|---|---|
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Xintao Wang et al., Tencent ARC Lab |
| [Real-ESRGAN ncnn-Vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | Vulkan inference binary |
| [4x-UltraSharp](https://huggingface.co/Kim2091/UltraSharp) | Kim2091 |
| [rembg](https://github.com/danielgatis/rembg) | Daniel Gatis |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Ultralytics |
| [BasicSR](https://github.com/xinntao/BasicSR) | Xintao Wang et al. |
| [tkinterdnd2](https://github.com/pmgagne/tkinterdnd2) | pmgagne |
| [OpenModelDB](https://openmodeldb.info) | Community model repository |

---

## License

MIT — see [LICENSE](LICENSE)
