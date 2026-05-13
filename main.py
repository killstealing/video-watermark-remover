"""
Video Watermark Remover (VWR) — CLI entry point.

Usage examples:
    python main.py --input video.mp4 --output clean.mp4 --coords 100 200 150 50
    python main.py --input https://example.com/video.mp4 --output clean.mp4 --text "Sample"
    python main.py --input clip.mkv --output out.mp4 --logo watermark.png
    python main.py --input video.mp4 --output clean.mp4 --coords 100 200 150 50 --method delogo
"""

import argparse
import os
import sys

import cv2

from downloader import download_video
from utils import (
    check_ffmpeg,
    clean_temp_dir,
    ensure_dir,
    is_url,
    log,
    setup_logging,
)
from video_io import apply_delogo_ffmpeg, get_video_info
from watermark_locator import locate_watermark, build_locator
from watermark_remover import (
    remove_watermark_inpaint_with_progress,
    remove_watermark_inpaint_dynamic,
    INPAINT_METHODS,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Watermark Remover — remove watermarks from video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Manual coordinates
  python main.py --input video.mp4 --output clean.mp4 --coords 100 200 150 50

  # Auto-detect text watermark
  python main.py --input video.mp4 --output clean.mp4 --text "Sample"

  # Auto-detect logo watermark
  python main.py --input clip.mkv --output out.mp4 --logo watermark.png

  # Use FFmpeg delogo filter (faster)
  python main.py --input video.mp4 --output clean.mp4 --coords 100 200 150 50 --method delogo
        """,
    )

    parser.add_argument("--input", "-i", required=True, help="Input video path or URL")
    parser.add_argument("--output", "-o", required=True, help="Output video path")

    wm_group = parser.add_argument_group("watermark location (choose one)")
    wm_group.add_argument("--coords", nargs=4, type=int, action="append", metavar=("X", "Y", "W", "H"),
                          help="Manual watermark rectangle (repeat for multiple regions)")
    wm_group.add_argument("--text", type=str, help="Watermark text to detect via OCR")
    wm_group.add_argument("--logo", type=str, help="Path to watermark logo/template image")

    parser.add_argument("--method", choices=["telea", "ns", "delogo"], default="telea",
                        help="Removal method (default: telea)")
    parser.add_argument("--corners", action="store_true",
                        help="Auto-detect and clean watermarks in top-left & bottom-right corners")
    parser.add_argument("--dynamic", action="store_true",
                        help="Re-detect watermark position on every frame (for moving watermarks)")
    parser.add_argument("--detect-interval", type=int, default=5, metavar="N",
                        help="When --dynamic with OCR, re-detect every N frames (default: 5)")
    parser.add_argument("--temp-dir", default="./temp", help="Temporary directory (default: ./temp)")
    parser.add_argument("--keep-audio", action="store_true", default=True,
                        help="Keep original audio track (default)")
    parser.add_argument("--no-audio", action="store_true", help="Discard audio track")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def _find_text_in_roi(roi_gray, pct=85):
    """Find text bounding box in a corner ROI via Laplacian edge projection."""
    import numpy as np

    lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
    lap_abs = np.abs(lap)

    h_proj = np.mean(lap_abs, axis=1)
    h_t = np.percentile(h_proj, pct)
    h_idx = np.where(h_proj > h_t)[0]
    if len(h_idx) < 6:
        return None

    y1, y2 = int(h_idx[0]), int(h_idx[-1])
    v_proj = np.mean(lap_abs[y1:y2, :], axis=0)
    v_t = np.percentile(v_proj, pct)
    v_idx = np.where(v_proj > v_t)[0]
    if len(v_idx) < 10:
        return None

    x1, x2 = int(v_idx[0]), int(v_idx[-1])
    rw, rh = x2 - x1, y2 - y1
    if rw < 30 or rh < 8 or rw > 350 or rh > 100 or rw / rh > 20 or rw / rh < 1.2:
        return None
    return (x1, y1, rw, rh)


def detect_corner_watermarks(video_path):
    """Auto-detect watermark positions in top-left and bottom-right corners."""
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Try multiple time points; keep best detection per corner by edge score
    check_times = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 9.0]
    best_per_corner = {}  # corner -> (x, y, w, h, score)

    for t in check_times:
        fn = min(int(t * fps), total_frames - 1)
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap2.read()
        cap2.release()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Bottom-right corner: slide window
        for y_off in [0, 20, 40, 60]:
            y1 = height - 160 + y_off
            y2 = y1 + 120
            if y2 > height:
                break
            br_roi = gray[y1:y2, width - 250:width]
            br_region = _find_text_in_roi(br_roi)
            if br_region:
                rx, ry, rw, rh = br_region
                # Edge score
                lap = cv2.Laplacian(br_roi[ry:ry + rh, rx:rx + rw], cv2.CV_64F)
                score = float(np.mean(np.abs(lap)))
                margin = 20
                fx = max(0, (width - 250) + rx - margin)
                fy = max(0, y1 + ry - margin)
                fw = min(width - fx, rw + margin * 2)
                fh = min(height - fy, rh + margin * 2)
                if "BR" not in best_per_corner or score > best_per_corner["BR"][4]:
                    best_per_corner["BR"] = (fx, fy, fw, fh, score)
                break

        # Top-left corner: slide window
        for y_off in [0, 20, 40, 60]:
            y1 = y_off
            y2 = y1 + 120
            if y2 > 200:
                break
            tl_roi = gray[y1:y2, 0:440]
            tl_region = _find_text_in_roi(tl_roi)
            if tl_region:
                rx, ry, rw, rh = tl_region
                lap = cv2.Laplacian(tl_roi[ry:ry + rh, rx:rx + rw], cv2.CV_64F)
                score = float(np.mean(np.abs(lap)))
                margin = 20
                fx = max(0, rx - margin)
                fy = max(0, y1 + ry - margin)
                fw = min(width - fx, rw + margin * 2)
                fh = min(height - fy, rh + margin * 2)
                if "TL" not in best_per_corner or score > best_per_corner["TL"][4]:
                    # TL watermarks may span a wide area — extend coverage
                    fw = min(width - fx, max(fw, 380))
                    fh = min(height - fy, max(fh, 100))
                    best_per_corner["TL"] = (fx, fy, fw, fh, score)
                break

    results = []
    for corner in ["BR", "TL"]:
        if corner in best_per_corner:
            x, y, w, h, score = best_per_corner[corner]
            results.append((x, y, w, h, corner, score))

    log.info("Auto-detected %d corner watermark(s)", len(results))
    for (x, y, w, h, corner, score) in results:
        log.info("  %s: x=%d y=%d w=%d h=%d (score=%.1f)", corner, x, y, w, h, score)

    return [(x, y, w, h) for (x, y, w, h, *_) in results]


def main():
    args = parse_args()

    if args.verbose:
        setup_logging(level=10)  # DEBUG
    else:
        setup_logging()

    check_ffmpeg()

    temp_dir = args.temp_dir
    ensure_dir(temp_dir)

    keep_audio = args.keep_audio and not args.no_audio

    # --- Step 1: obtain local video file ---
    if is_url(args.input):
        log.info("Input is a URL — downloading...")
        video_path = download_video(args.input, temp_dir)
    else:
        video_path = args.input
        if not os.path.isfile(video_path):
            log.error("Input file not found: %s", video_path)
            sys.exit(1)

    # --- Step 2: read video metadata ---
    info = get_video_info(video_path)
    log.info(
        "Video: %dx%d @ %.2f fps, codec=%s, audio=%s, frames=%s",
        info["width"], info["height"], info["fps"],
        info["codec"], info["has_audio"],
        info.get("total_frames", "unknown"),
    )

    # --- Step 3: determine watermark location strategy ---
    coords = [tuple(c) for c in args.coords] if args.coords else None
    template_path = args.logo
    text = args.text
    dynamic = args.dynamic

    if args.corners:
        coords = detect_corner_watermarks(video_path)
        if not coords:
            log.error("--corners: could not detect watermarks. "
                      "Try --coords manually or use --at with detect.py")
            sys.exit(1)
        dynamic = False

    if dynamic and coords:
        log.warning("--dynamic has no effect with --coords (coords are static); ignoring --dynamic")
        dynamic = False

    # Delogo path (static only — delogo can't do per-frame re-detection)
    if args.method == "delogo":
        if dynamic:
            log.error("--method delogo does not support --dynamic (moving watermarks)")
            sys.exit(1)
        if coords is None:
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()
            if not ret:
                log.error("Cannot read first frame from video")
                sys.exit(1)
            coords = [locate_watermark(first_frame, template_path, text, None)]
        apply_delogo_ffmpeg(video_path, args.output, *coords[0], keep_audio=keep_audio)
        log.info("Done. Output: %s", args.output)
        return

    # Inpaint path — static vs dynamic
    if dynamic:
        locator = build_locator(
            template_path=template_path, text=text,
            dynamic=True, detect_interval=args.detect_interval,
        )
        # Try to find watermark in first few frames
        cap = cv2.VideoCapture(video_path)
        found = False
        for attempt in range(min(30, int(info.get('total_frames', 240)))):
            ret, frame = cap.read()
            if not ret:
                break
            coords_check = locator(frame)
            if coords_check:
                found = True
                for c in coords_check:
                    log.info("Watermark first detected at frame %d: (%d,%d,%d,%d)", attempt, *c)
                break
        cap.release()
        if not found:
            log.error("Watermark detection failed on first 30 frames. "
                      "Try lowering the threshold or providing manual --coords.")
            sys.exit(1)
        log.info("Dynamic watermark detection active (mode: %s)", "template" if template_path else "ocr")
        remove_watermark_inpaint_dynamic(
            video_path, args.output, info, locator,
            method=args.method, temp_dir=temp_dir, keep_audio=keep_audio,
        )
    else:
        # Static path: either explicit --coords (single or multiple) or auto-detect
        if coords is not None:
            # User provided explicit coordinates
            coords_list = coords
            for c in coords_list:
                log.info("Watermark region: x=%d y=%d w=%d h=%d", *c)
        else:
            locator = build_locator(
                template_path=template_path, text=text, coords=None, dynamic=False,
            )
            # Get coords from first frame
            cap = cv2.VideoCapture(video_path)
            ret, first_frame = cap.read()
            cap.release()
            if not ret:
                log.error("Cannot read first frame from video")
                sys.exit(1)
            coords_list = locator(first_frame)
            if not coords_list:
                log.error("Static watermark detection failed")
                sys.exit(1)
            for c in coords_list:
                log.info("Watermark region: x=%d y=%d w=%d h=%d", *c)

        remove_watermark_inpaint_with_progress(
            video_path, args.output, coords_list, info,
            method=args.method, temp_dir=temp_dir, keep_audio=keep_audio,
        )

    # --- Cleanup: remove temp dir if we downloaded ---
    if is_url(args.input):
        clean_temp_dir(temp_dir)

    log.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
