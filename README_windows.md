# Windows Quick Reference

## Portable EXE (recommended)

```
1. Download AI.Image.Upscaler.v1.2.0.zip from the Releases page
2. Extract — keep all files in the same folder
3. Run AI Image Upscaler.exe
```

The `models/`, `realesrgan-ncnn-vulkan.exe`, and `_internal/` folders must stay
next to the EXE. Do not move the EXE out of its folder.

---

## Running from source

```powershell
python -m pip install -r requirements.txt
python main.py
```

---

## NCNN binary — CLI usage

`realesrgan-ncnn-vulkan.exe` can also be used directly:

```
realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png [options]

  -i  input image or folder
  -o  output image or folder
  -n  model name (default: realesrgan-x4plus)
  -s  scale: 2, 3, or 4  (default: 4)
  -m  model folder path  (default: models/)
  -t  tile size  (0 = auto, 32 = sharper on AMD)
  -g  GPU ID  (default: auto)
  -j  threads  load:proc:save  (default: 2:2:2)
  -x  enable TTA mode
  -f  output format: jpg / png / webp
  -v  verbose output
```

### Examples

```powershell
# General photo
.\realesrgan-ncnn-vulkan.exe -i photo.jpg -o photo_4x.png -n realesrgan-x4plus -m models

# AI art on AMD GPU (fp32 avoids fp16 precision issues)
.\realesrgan-ncnn-vulkan.exe -i art.jpg -o art_4x.png -n 4x-UltraSharp-fp32 -s 4 -m models -t 32

# Anime video frames at 4x
.\realesrgan-ncnn-vulkan.exe -i tmp_frames -o out_frames -n realesr-animevideov3 -s 4 -m models -f jpg

# Batch folder
.\realesrgan-ncnn-vulkan.exe -i input_folder -o output_folder -n realesrgan-x4plus -m models -f png
```

---

## Video upscaling workflow

```powershell
# Step 1 — extract frames
mkdir tmp_frames
ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 tmp_frames/frame%08d.jpg

# Step 2 — upscale  (or use the GUI with realesr-animevideov3-x4)
mkdir out_frames
.\realesrgan-ncnn-vulkan.exe -i tmp_frames -o out_frames -n realesr-animevideov3 -s 4 -m models -f jpg

# Step 3 — merge with original audio
ffmpeg -i out_frames/frame%08d.jpg -i video.mp4 -map 0:v:0 -map 1:a:0 `
       -c:a copy -c:v libx264 -r 23.98 -pix_fmt yuv420p output_upscaled.mp4
```

---

## Building a standalone EXE

```powershell
# Create a minimal build venv (important — avoids bloating the EXE)
python -m venv venv_build
.\venv_build\Scripts\pip install Pillow tkinterdnd2 opencv-python-headless "rembg[cpu]" ultralytics piexif pyinstaller

# Build
.\venv_build\Scripts\python -m PyInstaller main.spec --noconfirm

# Copy models + binary into the dist folder
python post_build.py

# Output: dist\AI Image Upscaler\AI Image Upscaler.exe  (~10 MB launcher)
```

> **Why venv_build?** The system Python may have torch/scipy installed, which bloats  
> the EXE from ~60 MB to 200+ MB. The build venv contains only runtime deps.

---

## GPU driver links

- **AMD:** https://www.amd.com/en/support
- **NVIDIA:** https://www.nvidia.com/Download/index.aspx
- **Intel:** https://www.intel.com/content/www/us/en/download-center/home.html

---

For full documentation see [README.md](README.md)
