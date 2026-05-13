import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


def setup_logging(level=logging.INFO):
    logger = logging.getLogger("vwr")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


log = setup_logging()


def check_ffmpeg():
    """Verify ffmpeg is available on PATH."""
    if not shutil.which("ffmpeg"):
        log.error(
            "ffmpeg not found on PATH. Install it from https://ffmpeg.org/download.html"
        )
        sys.exit(1)
    log.info("ffmpeg found: %s", shutil.which("ffmpeg"))


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_coords(x, y, w, h, frame_width, frame_height):
    """Check that the watermark rectangle lies within the frame bounds."""
    if x < 0 or y < 0:
        raise ValueError(f"Coordinates ({x}, {y}) must be non-negative")
    if x + w > frame_width or y + h > frame_height:
        raise ValueError(
            f"Watermark region ({x},{y},{w},{h}) exceeds frame size "
            f"({frame_width}x{frame_height})"
        )
    if w <= 0 or h <= 0:
        raise ValueError("Watermark width and height must be positive")


def is_url(path):
    """Return True if path looks like a URL."""
    return path.startswith(("http://", "https://", "ftp://"))


def get_video_info_ffprobe(video_path):
    """Use ffprobe to extract width, height, fps, total frames, codec, and audio streams."""
    import json

    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream = None
    audio_stream = None
    for stream in info.get("streams", []):
        if stream["codec_type"] == "video" and video_stream is None:
            video_stream = stream
        elif stream["codec_type"] == "audio":
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError("No video stream found")

    width = video_stream["width"]
    height = video_stream["height"]
    codec = video_stream.get("codec_name", "unknown")

    # Determine FPS
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0

    nb_frames = video_stream.get("nb_frames")
    if not nb_frames:
        nb_frames = int(info.get("format", {}).get("nb_streams", 0))
        if nb_frames == 0:
            nb_frames = None  # unknown

    has_audio = audio_stream is not None

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "codec": codec,
        "total_frames": nb_frames,
        "has_audio": has_audio,
    }


def clean_temp_dir(temp_dir):
    """Remove the temporary directory and all contents."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        log.info("Cleaned up temp directory: %s", temp_dir)
