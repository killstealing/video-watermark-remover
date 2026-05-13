import json
import os
import subprocess

import numpy as np

from utils import ensure_dir, log


def get_video_info(video_path):
    """Extract video metadata via ffprobe.

    Returns a dict with keys: width, height, fps, codec, total_frames, has_audio.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path,
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
        raise RuntimeError("No video stream found in input")

    # FPS
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0

    # Total frames
    nb_frames = video_stream.get("nb_frames")
    if nb_frames:
        nb_frames = int(nb_frames)
    else:
        nb_frames_str = info.get("format", {}).get("nb_streams")
        nb_frames = int(nb_frames_str) if nb_frames_str else None

    return {
        "width": video_stream["width"],
        "height": video_stream["height"],
        "fps": fps,
        "codec": video_stream.get("codec_name", "h264"),
        "total_frames": nb_frames,
        "has_audio": audio_stream is not None,
    }


def read_frames_raw(video_path):
    """Generator that yields raw BGR frames from a video using FFmpeg.

    Yields (frame_index, frame) where frame is a numpy array (H, W, 3).
    """
    info = get_video_info(video_path)
    w, h = info["width"], info["height"]

    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-an", "-sn",       # no audio, no subtitles
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = w * h * 3
    idx = 0

    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        yield idx, frame
        idx += 1

    proc.stdout.close()
    proc.wait()


def count_frames(video_path):
    """Return the total number of frames in a video (fast count via ffprobe streams)."""
    info = get_video_info(video_path)
    if info["total_frames"] is not None:
        return info["total_frames"]

    # Fallback: count by parsing ffprobe packet info
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def write_video_inpaint(
    output_path, input_path, mask, inpaint_fn, info, temp_dir, keep_audio=True
):
    """Write the output video by reading frames, applying *inpaint_fn*, and
    piping them to ffmpeg for encoding.

    This function avoids keeping all frames in memory — it streams one frame
    at a time.

    Args:
        output_path: destination file path.
        input_path: original video (for audio extraction if *keep_audio*).
        mask: single-channel uint8 mask (same size as frames).
        inpaint_fn: callable(frame, mask) → inpainted_frame.
        info: dict from get_video_info().
        temp_dir: directory for temporary files.
        keep_audio: whether to copy the original audio track.
    """
    ensure_dir(temp_dir)
    w, h, fps = info["width"], info["height"], info["fps"]

    # Build ffmpeg encoder command
    # Pipe raw BGR24 frames to ffmpeg; encode with libx264.
    encoder_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",            # stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
    ]

    if keep_audio and info["has_audio"]:
        # Mux audio from the original file
        encoder_cmd += ["-i", input_path, "-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0"]
    else:
        encoder_cmd += ["-an"]

    encoder_cmd.append(output_path)

    log.info("Encoding output video: %s", output_path)
    proc = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Encode frames
    for idx, frame in read_frames_raw(input_path):
        repaired = inpaint_fn(frame, mask)
        proc.stdin.write(repaired.tobytes())

    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode(errors='replace')}")

    log.info("Video written successfully: %s", output_path)


def apply_delogo_ffmpeg(input_path, output_path, x, y, w, h, keep_audio=True):
    """Use FFmpeg's built-in delogo filter (fast but limited quality)."""
    log.info("Applying FFmpeg delogo filter at (%d,%d,%d,%d)", x, y, w, h)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"delogo=x={x}:y={y}:w={w}:h={h}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
    ]
    if keep_audio:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-an"]

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg delogo failed: {result.stderr}")
    log.info("delogo video written: %s", output_path)
