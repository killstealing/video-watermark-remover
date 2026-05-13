"""
Step 1: Detect watermark positions from user-specified time points.

Uses Laplacian edge projection in small corner windows to find
watermark text boundaries, then cross-validates via template matching.

Usage:
    python detect.py --input video.mp4 --at 2s 5s
"""
import argparse
import sys
import cv2
import numpy as np


def parse_time(time_str):
    time_str = time_str.strip().lower()
    if time_str.endswith("ms"):
        return float(time_str[:-2]) / 1000.0
    if time_str.endswith("s"):
        return float(time_str[:-1])
    return float(time_str)


def find_text_region(roi_gray):
    """Find the bounding box of text within a small corner ROI.

    Uses Laplacian edge projection: text strokes produce strong edges
    that cluster together along horizontal and vertical axes.

    Returns (rx, ry, rw, rh) relative to roi_gray, or None.
    """
    h, w = roi_gray.shape

    # Laplacian edge detection
    lap = cv2.Laplacian(roi_gray, cv2.CV_64F)
    lap_abs = np.abs(lap)

    # Horizontal projection: which rows have text?
    h_proj = np.mean(lap_abs, axis=1)
    h_thresh = np.percentile(h_proj, 65)
    h_mask = h_proj > h_thresh
    h_indices = np.where(h_mask)[0]

    if len(h_indices) < 8:
        return None

    # Find the main cluster of text rows
    y1 = int(h_indices[0])
    y2 = int(h_indices[-1])

    # Vertical projection: which columns have text?
    v_proj = np.mean(lap_abs[y1:y2, :], axis=0)
    v_thresh = np.percentile(v_proj, 65)
    v_mask = v_proj > v_thresh
    v_indices = np.where(v_mask)[0]

    if len(v_indices) < 15:
        return None

    x1 = int(v_indices[0])
    x2 = int(v_indices[-1])

    rw = x2 - x1
    rh = y2 - y1

    # Sanity checks: text-like size and aspect
    if rw < 30 or rh < 8:
        return None
    if rw > 300 or rh > 100:
        return None
    if rw / rh > 20 or rw / rh < 1.5:
        return None

    return (x1, y1, rw, rh)


def find_watermarks_at_times(video_path, time_points, margin=15):
    """Find watermark positions by analyzing frames at given time points.

    For each time point:
      1. Scan small corner windows with Laplacian edge projection
      2. Find text-like regions → candidate watermark positions
      3. Extract templates from the best candidate per corner
      4. Cross-validate templates by matching against other time points

    Returns list of (x, y, w, h, corner, score) for confirmed positions.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {width}x{height}, {total_frames} frames, {fps:.0f} fps")

    # Small focused windows in each corner (where watermarks typically sit)
    # (y1, y2, x1, x2) — tight windows, not the whole corner
    corner_windows = {
        "BR": (height - 160, height, width - 250, width),
        "TL": (0, 180, 0, 380),
        "TR": (0, 180, width - 380, width),
        "BL": (height - 160, height, 0, 380),
    }

    # --- Phase 1: find candidates at each time point ---
    candidates = []  # (x, y, w, h, corner, time_sec, template_gray, edge_score)

    for t in time_points:
        fn = min(int(t * fps), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            print(f"  {t:.1f}s: frame read failed")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found_at_t = []

        for corner, (y1, y2, x1, x2) in corner_windows.items():
            roi = gray[y1:y2, x1:x2]
            region = find_text_region(roi)

            if region is None:
                continue

            rx, ry, rw, rh = region

            # Extract as template (tight crop)
            template = roi[ry:ry + rh, rx:rx + rw]

            # Edge score
            lap = cv2.Laplacian(template, cv2.CV_64F)
            edge_score = float(np.mean(np.abs(lap)))

            # Full-frame coordinates
            fx = x1 + rx - margin
            fy = y1 + ry - margin
            fw = rw + margin * 2
            fh = rh + margin * 2
            fx = max(0, fx)
            fy = max(0, fy)
            fw = min(width - fx, fw)
            fh = min(height - fy, fh)

            found_at_t.append((fx, fy, fw, fh, corner, t, template, edge_score))

        # Keep top 2 per time point (different corners preferred)
        found_at_t.sort(key=lambda c: c[7], reverse=True)
        seen_corners = set()
        top = []
        for c in found_at_t:
            if c[4] not in seen_corners:
                top.append(c)
                seen_corners.add(c[4])
            if len(top) >= 2:
                break

        desc = ", ".join(f"({c[0]},{c[1]} {c[2]}x{c[3]} [{c[4]}] s={c[7]:.1f})"
                        for c in top)
        print(f"  {t:.1f}s: {desc}")
        candidates.extend(top)

    if not candidates:
        print("\nNo watermark candidates found.")
        return []

    # --- Phase 2: cross-validate templates via template matching ---
    # For each unique template, match against ALL time points.
    # A real watermark will match at the SAME position across multiple frames.
    confirmed = {}
    cap2 = cv2.VideoCapture(video_path)

    # Deduplicate similar templates
    unique_templates = []
    for c in candidates:
        fx, fy, fw, fh, corner, t_sec, tmpl, score = c
        th, tw = tmpl.shape
        dup = False
        for ufx, ufy, ufw, ufh, ucorner, utmpl, uscore in unique_templates:
            if abs(tw - utmpl.shape[1]) < 20 and abs(th - utmpl.shape[0]) < 10:
                dup = True
                break
        if not dup:
            unique_templates.append((fx, fy, fw, fh, corner, tmpl, score))

    print(f"\nCross-validating {len(unique_templates)} unique templates...")

    for fx, fy, fw, fh, corner, tmpl, _ in unique_templates:
        th, tw = tmpl.shape
        # Match this template against all time points
        matches_at_positions = {}  # quantized_position -> [scores]

        for t in time_points:
            fn = min(int(t * fps), total_frames - 1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap2.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)

            # Find all peaks above 0.7 in corner regions
            ys, xs = np.where(result >= 0.7)
            for cy, cx in zip(ys, xs):
                score = float(result[cy, cx])
                # Quantize position
                qx = (cx // 20) * 20
                qy = (cy // 20) * 20
                key = (qx, qy)
                if key not in matches_at_positions:
                    matches_at_positions[key] = []
                matches_at_positions[key].append((score, cx, cy))

        # Find the best position for this template
        if not matches_at_positions:
            continue

        best_pos = max(matches_at_positions.items(),
                       key=lambda item: (len(item[1]), np.mean([s[0] for s in item[1]])))

        (qx, qy), match_list = best_pos
        avg_score = np.mean([s[0] for s in match_list])
        match_count = len(match_list)
        mx = int(np.median([s[1] for s in match_list]))
        my = int(np.median([s[2] for s in match_list]))

        # Only keep if matches at multiple time points (persistent watermark)
        if match_count < 2 or avg_score < 0.72:
            continue

        key = (mx // 30 * 30, my // 30 * 30)
        if key not in confirmed or avg_score > confirmed[key][5]:
            confirmed[key] = (mx - margin, my - margin, tw + margin * 2,
                              th + margin * 2, corner, avg_score, match_count)

    cap2.release()

    # --- Phase 3: output ---
    if not confirmed:
        print("No confirmed watermark positions (template validation failed).")
        return []

    results = []
    for (fx, fy, fw, fh, corner, score, count) in confirmed.values():
        fx = max(0, fx)
        fy = max(0, fy)
        fw = min(width - fx, fw)
        fh = min(height - fy, fh)
        results.append((fx, fy, fw, fh, corner, score, count))

    results.sort(key=lambda r: r[4])  # sort by corner name

    print(f"\n{'Corner':<6} {'x':>5} {'y':>5} {'w':>5} {'h':>5}  "
          f"{'score':>7}  {'matches':>7}")
    print("-" * 50)
    for (x, y, w, h, corner, score, count) in results:
        print(f"{corner:<6} {x:>5} {y:>5} {w:>5} {h:>5}  "
              f"{score:>7.3f}  {count:>7}")

    return [(x, y, w, h, corner, score) for (x, y, w, h, corner, score, _)
            in results]


def main():
    parser = argparse.ArgumentParser(
        description="Detect watermark positions from time points")
    parser.add_argument("--input", "-i", required=True,
                        help="Input video path")
    parser.add_argument("--at", nargs="+", required=True,
                        help="Time points, e.g. --at 2s 5s")
    parser.add_argument("--margin", type=int, default=15,
                        help="Padding around detected region (default: 15px)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    times = [parse_time(t) for t in args.at]
    print(f"Input: {args.input}")
    print(f"Time points: {[f'{t:.1f}s' for t in times]}\n")

    results = find_watermarks_at_times(args.input, times, args.margin)

    if not results:
        print("\nNo watermarks detected.")
        sys.exit(1)

    print(f"\n=== {len(results)} watermark position(s) ===")

    # Output command
    coords_str = " ".join(
        f"--coords {x} {y} {w} {h}" for (x, y, w, h, *_) in results)
    print(f"\nStep 2 — remove with:")
    print(f"  python main.py --input {args.input} "
          f"--output clean.mp4 {coords_str}")

    if args.json:
        import json
        print(json.dumps([{"x": x, "y": y, "w": w, "h": h, "corner": c}
                          for (x, y, w, h, c, *_) in results]))


if __name__ == "__main__":
    main()
