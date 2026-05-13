# Video Watermark Remover (VWR)

Remove watermarks from video files — supports manual coordinates, template matching, and OCR-based text detection.

## Requirements

- **Python 3.8+**
- **FFmpeg** — must be installed separately and available on PATH.
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## Installation

```bash
pip install -r requirements.txt
```

### Optional: OCR-based text watermark detection

If you want to use `--text` for automatic text watermark detection:

```bash
pip install paddlepaddle paddleocr
```

## Usage

### Manual coordinates (static watermark)

```bash
python main.py --input video.mp4 --output clean.mp4 --coords 100 200 150 50
```

### Template matching (static logo watermark)

```bash
python main.py --input video.mp4 --output clean.mp4 --logo watermark.png
```

### Moving watermark — per-frame template matching

```bash
python main.py --input video.mp4 --output clean.mp4 --logo watermark.png --dynamic
```

### Moving watermark — OCR text detection (re-detect every 10 frames)

```bash
python main.py --input video.mp4 --output clean.mp4 --text "SampleText" --dynamic --detect-interval 10
```

### OCR text detection (static)

```bash
python main.py --input video.mp4 --output clean.mp4 --text "SampleText"
```

### Network video download

```bash
python main.py --input "https://example.com/video.mp4" --output clean.mp4 --coords 100 200 150 50
```

### FFmpeg delogo filter (faster, simpler)

```bash
python main.py --input video.mp4 --output clean.mp4 --coords 100 200 150 50 --method delogo
```

### Discard audio

```bash
python main.py --input video.mp4 --output clean.mp4 --coords 100 200 150 50 --no-audio
```

## Options

| Flag | Description |
|---|---|
| `--input`, `-i` | Input video path or URL |
| `--output`, `-o` | Output video path |
| `--coords X Y W H` | Manual watermark rectangle (x, y, width, height) |
| `--text TEXT` | Watermark text to detect via OCR |
| `--logo PATH` | Watermark template image for matching |
| `--method {telea,ns,delogo}` | Removal algorithm (default: `telea`) |
| `--dynamic` | Per-frame re-detection for moving watermarks |
| `--detect-interval N` | With `--dynamic` + OCR, re-detect every N frames (default: 5) |
| `--temp-dir PATH` | Temporary directory (default: `./temp`) |
| `--keep-audio` | Keep original audio (default) |
| `--no-audio` | Discard audio track |
| `--verbose`, `-v` | Verbose logging |

## How It Works

1. **Input**: local file or URL (auto-downloaded via yt-dlp).
2. **Locate**: the watermark is found by explicit coordinates, template matching, or OCR.
   - Static mode (default): detect once on the first frame, reuse the same mask for all frames.
   - Dynamic mode (`--dynamic`): re-detect on every frame (template matching) or every N frames (OCR), suitable for moving watermarks.
3. **Mask**: a binary mask is created covering the watermark region.
4. **Inpaint**: each frame is processed with OpenCV's `cv2.inpaint` (Telea or Navier-Stokes), then re-encoded via FFmpeg.
5. **Output**: video with the original resolution, frame rate, and audio track.

## Project Structure

```
vwr/
├── main.py                # CLI entry point
├── downloader.py          # Network video download (yt-dlp)
├── watermark_locator.py   # Watermark detection (template/OCR)
├── watermark_remover.py   # Inpainting logic
├── video_io.py            # Video read/write/FFmpeg wrappers
├── utils.py               # Helpers, logging, FFmpeg detection
├── requirements.txt
└── README.md
```


python main.py --input video1.mp4 --output clean.mp4 --text "豆包AI生成" --dynamic --detect-interval 10

python main.py --input video1.mp4 --output clean.mp4 --coords 200 1200 300 60

# 提取水印模板
python auto_detect.py --input video.mp4

# 动态去除
python main.py --input video.mp4 --output clean.mp4 --logo template.png --dynamic


# 2.0
# 自动检测水印并提取模板
python auto_detect.py --input video.mp4

# 动态去除（已有模板 template_v2.png）
python main.py --input video.mp4 --output clean.mp4 --logo template_v2.png --dynamic


# version 3.0
python main.py --input video1.mp4 --output clean.mp4 --logo template_v2.png --dynamic

# version 4.0
python main.py --input video1.mp4 --output clean.mp4 --logo template_v2.png --dynamic

# version 5.0
python main.py --input video1.mp4 --output clean.mp4 --coords 520 1180 200 80 --coords 30 20 360 130
