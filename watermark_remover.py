import cv2
import numpy as np
import threading
from tqdm import tqdm

from utils import log
from video_io import read_frames_raw, get_video_info, write_video_inpaint, apply_delogo_ffmpeg, count_frames


def create_mask(width, height, x, y, w, h, padding=6, feather=15):
    """Create a feathered mask for natural-looking inpainting.

    The mask edge transitions smoothly from 255 (full inpaint) to 0
    (no inpaint) over *feather* pixels, avoiding hard visible boundaries.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Inner rectangle: full inpaint strength
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    if x2 <= x1 or y2 <= y1:
        return mask

    mask[y1:y2, x1:x2] = 255

    # Apply Gaussian blur to feather the edges
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather / 2)

    return mask


def create_precise_mask(frame, coords, padding=20):
    """Create a hybrid mask: rectangle base + precise text detection.

    Uses the full rectangle as a safety net, plus additional edge/brightness
    detection to ensure semi-transparent text is fully covered.
    """
    x, y, w, h = coords
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros((height, width), dtype=np.uint8)

    # Base: rectangle mask (guaranteed coverage)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + w + padding)
    y2 = min(height, y + h + padding)

    if x2 <= x1 or y2 <= y1:
        return mask

    mask[y1:y2, x1:x2] = 255

    # Find locally bright pixels (semi-transparent white text + glow effects)
    roi = gray[y1:y2, x1:x2]

    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    background = cv2.morphologyEx(roi, cv2.MORPH_OPEN, bg_kernel)
    diff = cv2.subtract(roi, background)

    # Use a very low threshold to catch subtle glow/halo around text
    _, bright_mask = cv2.threshold(diff, 6, 255, cv2.THRESH_BINARY)

    # Heavy dilation to expand coverage to include animation glow/shadow
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=4)

    # Also expand rectangle mask by dilating to catch any animated halo
    rect_mask = mask[y1:y2, x1:x2].copy()
    rect_mask = cv2.dilate(rect_mask, kernel, iterations=2)

    # Combine rectangle + bright detection
    mask[y1:y2, x1:x2] = cv2.bitwise_or(rect_mask, bright_mask)

    return mask


def inpaint_telea(frame, mask):
    """Inpaint a single frame using the Telea algorithm."""
    return cv2.inpaint(frame, mask, inpaintRadius=8, flags=cv2.INPAINT_TELEA)


def inpaint_ns(frame, mask):
    """Inpaint a single frame using the Navier-Stokes algorithm."""
    return cv2.inpaint(frame, mask, inpaintRadius=8, flags=cv2.INPAINT_NS)


INPAINT_METHODS = {
    "telea": inpaint_telea,
    "ns": inpaint_ns,
}


def combine_masks(masks):
    """Combine multiple binary masks into one (logical OR)."""
    if not masks:
        return None
    result = masks[0].copy()
    for m in masks[1:]:
        result = cv2.bitwise_or(result, m)
    return result


def remove_watermark_inpaint(input_path, output_path, coords, info, method="telea", temp_dir="./temp", keep_audio=True):
    """Remove the watermark via per-frame OpenCV inpainting.

    Args:
        input_path: original video file.
        output_path: destination path.
        coords: (x, y, w, h) of the watermark.
        info: video metadata dict from get_video_info().
        method: 'telea' or 'ns'.
        temp_dir: directory for temporary work files.
        keep_audio: whether to copy the original audio track.
    """
    x, y, w, h = coords
    width, height = info["width"], info["height"]

    mask = create_mask(width, height, x, y, w, h)
    inpaint_fn = INPAINT_METHODS.get(method, inpaint_telea)
    log.info("Using inpainting method: %s", method)

    total = count_frames(input_path)
    if total is None:
        total = info.get("total_frames")

    write_video_inpaint(output_path, input_path, mask, inpaint_fn, info, temp_dir, keep_audio)

    # Show progress retroactively — the streaming encode already happened.
    # For a real progress bar we'd need to count before encoding, which we do
    # via count_frames above. The actual bar is integrated directly below
    # when we re-run with progress. Let's build an integrated version.
    log.info("Inpainting complete.")


def remove_watermark_inpaint_with_progress(
    input_path, output_path, coords, info, method="telea", temp_dir="./temp", keep_audio=True
):
    """Inpaint with static mask(s) — supports single coords (x,y,w,h) or list of coords."""
    if isinstance(coords, tuple):
        coords_list = [coords]
    else:
        coords_list = coords

    def _static_mask_fn(frame):
        masks = [create_mask(info["width"], info["height"], x, y, w, h)
                 for (x, y, w, h) in coords_list]
        if len(masks) == 1:
            return masks[0]
        return combine_masks(masks)

    _encode_inpaint(input_path, output_path, info, _static_mask_fn, method, temp_dir, keep_audio)


def remove_watermark_inpaint_dynamic(
    input_path, output_path, info, locator, method="telea", temp_dir="./temp", keep_audio=True
):
    """Inpaint with per-frame watermark detection.

    Args:
        locator: callable(frame) → (x, y, w, h) — called on every frame.
    """
    width, height = info["width"], info["height"]

    def _dynamic_mask_fn(frame):
        coords_list = locator(frame)
        if not coords_list:
            return None

        confidences = locator.last_confidences if locator.last_confidences else [0.7] * len(coords_list)

        masks = []
        for i, coords in enumerate(coords_list):
            confidence = confidences[i] if i < len(confidences) else 0.7

            # Adaptive padding: lower confidence = animation/poor match = larger mask
            if confidence > 0.85:
                padding = 20
            elif confidence > 0.75:
                padding = 28
            else:
                padding = 36

            masks.append(create_precise_mask(frame, coords, padding=padding))

        if len(masks) == 1:
            return masks[0]
        return combine_masks(masks)

    _encode_inpaint(input_path, output_path, info, _dynamic_mask_fn, method, temp_dir, keep_audio)


def _encode_inpaint(input_path, output_path, info, mask_fn, method, temp_dir, keep_audio):
    """Core encoding loop: for each frame, call mask_fn to get a mask, inpaint, and write."""
    import subprocess
    from utils import ensure_dir

    width, height = info["width"], info["height"]
    fps = info["fps"]
    inpaint_fn = INPAINT_METHODS.get(method, inpaint_telea)
    total = count_frames(input_path)
    ensure_dir(temp_dir)

    encoder_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
    ]

    if keep_audio and info["has_audio"]:
        encoder_cmd += ["-i", input_path]
    else:
        encoder_cmd += ["-an"]

    encoder_cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
    ]

    if keep_audio and info["has_audio"]:
        encoder_cmd += ["-c:a", "copy", "-map", "0:v:0", "-map", "1:a:0"]

    encoder_cmd.append(output_path)

    log.info("Encoding output video: %s", output_path)
    proc = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read stderr in a thread to prevent pipe buffer from blocking ffmpeg
    stderr_chunks = []
    def _read_stderr():
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    try:
        pbar = tqdm(total=total, unit="frame", desc="Inpainting")
        for _idx, frame in read_frames_raw(input_path):
            mask = mask_fn(frame)
            if mask is None:
                proc.stdin.write(frame.tobytes())
            else:
                repaired = inpaint_fn(frame, mask)
                proc.stdin.write(repaired.tobytes())
            pbar.update(1)
        pbar.close()
    finally:
        proc.stdin.close()
        proc.wait()
        stderr_thread.join(timeout=5)

    if proc.returncode != 0:
        stderr = b"".join(stderr_chunks)
        raise RuntimeError(f"ffmpeg encoding failed: {stderr.decode(errors='replace')}")

    log.info("Video written successfully: %s", output_path)
