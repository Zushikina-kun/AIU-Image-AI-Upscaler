# Windows Quick Reference

## Running the app

```powershell
python -m pip install -r requirements.txt
python main.py
```

## NCNN binary usage (CLI)

The included `realesrgan-ncnn-vulkan.exe` can also be used directly from the command line:

```
realesrgan-ncnn-vulkan.exe -i input.jpg -o output.png [options]

  -i  input image or folder
  -o  output image or folder
  -n  model name (default: realesrgan-x4plus)
  -s  scale: 2, 3, or 4 (default: 4)
  -m  model folder path (default: models/)
  -t  tile size (0 = auto)
  -g  GPU ID (default: auto)
  -j  threads load:proc:save (default: 2:2:2)
  -x  enable TTA mode
  -f  output format: jpg / png / webp
  -v  verbose output
```

### Example commands

```powershell
# Upscale single image
.\realesrgan-ncnn-vulkan.exe -i photo.jpg -o photo_4x.png -n realesrgan-x4plus -m models

# Upscale with UltraSharp fp32 (recommended for AMD RX 580)
.\realesrgan-ncnn-vulkan.exe -i photo.jpg -o photo_4x.png -n 4x-UltraSharp-fp32 -s 4 -m models

# Batch upscale a folder
.\realesrgan-ncnn-vulkan.exe -i input_folder -o output_folder -n realesrgan-x4plus -m models -f jpg

# Anime video frames at 2x
.\realesrgan-ncnn-vulkan.exe -i tmp_frames -o out_frames -n realesr-animevideov3 -s 2 -m models -f jpg
```

## Video upscaling workflow

```powershell
# Step 1 — extract frames
mkdir tmp_frames
ffmpeg -i video.mp4 -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 tmp_frames/frame%08d.jpg

# Step 2 — upscale frames (use the GUI or CLI)
mkdir out_frames
.\realesrgan-ncnn-vulkan.exe -i tmp_frames -o out_frames -n realesr-animevideov3 -s 2 -m models -f jpg

# Step 3 — merge back with original audio
ffmpeg -i out_frames/frame%08d.jpg -i video.mp4 -map 0:v:0 -map 1:a:0 `
       -c:a copy -c:v libx264 -r 23.98 -pix_fmt yuv420p output_upscaled.mp4
```

## GPU driver links (if you hit crashes)

- **AMD:** https://www.amd.com/en/support
- **NVIDIA:** https://www.nvidia.com/Download/index.aspx
- **Intel:** https://www.intel.com/content/www/us/en/download-center/home.html

## Building a standalone EXE

```powershell
pip install pyinstaller
pyinstaller main.spec
# Output: dist\AI Image Upscaler.exe
```

---

For full documentation see [README.md](README.md)
