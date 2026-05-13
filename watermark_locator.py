import cv2
import numpy as np

from utils import log, validate_coords


DEFAULT_TEMPLATE_THRESHOLD = 0.65
DEFAULT_DETECT_INTERVAL = 5  # frames between re-detection for slow methods (OCR)


class DynamicLocator:
    """Callable that locates watermarks on a per-frame basis.

    Caches expensive resources (template image, OCR instance) and
    falls back to the last-known-good position when detection fails.
    Supports detecting MULTIPLE simultaneous watermarks.
    """

    def __init__(self, mode, template_path=None, text=None, threshold=DEFAULT_TEMPLATE_THRESHOLD, detect_interval=1):
        self.mode = mode          # "template" | "ocr" | "static"
        self.threshold = threshold
        self.detect_interval = detect_interval
        self.last_coords_list = []  # list of (x, y, w, h)
        self.last_confidences = []  # list of float, parallel to last_coords_list
        self.frame_count = 0

        if mode == "template":
            if template_path is None:
                raise ValueError("template_path required for template mode")
            template = cv2.imread(template_path)
            if template is None:
                raise FileNotFoundError(f"Cannot read template image: {template_path}")
            self.template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            self.template_h, self.template_w = self.template_gray.shape

        elif mode == "ocr":
            if text is None:
                raise ValueError("text required for OCR mode")
            self.text = text
            self._ocr = None  # lazy init

    def _get_ocr(self):
        if self._ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        return self._ocr

    def __call__(self, frame):
        """Return list of watermark coordinates [(x,y,w,h), ...], or empty list."""
        self.frame_count += 1

        # Skip detection for intermediate frames (saves time with OCR)
        if self.mode != "static" and self.detect_interval > 1:
            if (self.frame_count - 1) % self.detect_interval != 0 and self.last_coords_list:
                return self.last_coords_list

        coords_list = []

        coords_list = []
        confidences = []

        if self.mode == "template":
            coords_list, confidences = self._match_template_multi(frame)
        elif self.mode == "ocr":
            coords = self._match_ocr(frame)
            if coords is not None:
                coords_list = [coords]
                confidences = [1.0]

        if coords_list:
            self.last_coords_list = coords_list
            self.last_confidences = confidences
        elif self.last_coords_list:
            # Detection failed this frame — reuse last known positions
            coords_list = self.last_coords_list
        # else: no detection and no history — return empty list

        return coords_list

    def _match_template_multi(self, frame):
        """Find ALL template matches above threshold.

        Returns (coords_list, confidences) where coords_list is [(x,y,w,h), ...]
        and confidences is a parallel list of correlation scores.

        Uses non-maximum suppression to avoid overlapping detections
        of the same watermark instance.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, self.template_gray, cv2.TM_CCOEFF_NORMED)

        # Find all peaks above threshold
        ys, xs = np.where(result >= self.threshold)
        if len(ys) == 0:
            return [], []

        # Sort by correlation (highest first)
        scores = result[ys, xs]
        order = np.argsort(scores)[::-1]
        xs_sorted = xs[order]
        ys_sorted = ys[order]

        # Non-maximum suppression: keep peaks that don't overlap
        kept = []
        kept_scores = []
        min_distance = max(self.template_w, self.template_h) // 2

        for i in range(len(xs_sorted)):
            cx, cy = xs_sorted[i], ys_sorted[i]
            # Check overlap with already-kept detections
            overlap = False
            for kx, ky, _, _ in kept:
                if abs(cx - kx) < min_distance and abs(cy - ky) < min_distance:
                    overlap = True
                    break
            if not overlap:
                kept.append((cx, cy, self.template_w, self.template_h))
                kept_scores.append(float(scores[order[i]]))

        # Apply position consistency
        if self.last_coords_list and kept:
            result_list = list(kept)
            result_scores = list(kept_scores)

            for old_coords, old_conf in zip(self.last_coords_list, self.last_confidences):
                lx, ly, lw, lh = old_coords
                matched = False
                for nx, ny, nw, nh in kept:
                    if abs(nx - lx) < min_distance and abs(ny - ly) < min_distance:
                        matched = True
                        break
                if not matched:
                    old_y = max(0, ly - 5)
                    old_x = max(0, lx - 5)
                    old_h = min(result.shape[0] - old_y, lh + 10)
                    old_w = min(result.shape[1] - old_x, lw + 10)
                    if old_h > 0 and old_w > 0:
                        old_region = result[old_y:old_y + old_h, old_x:old_x + old_w]
                        old_best = float(np.max(old_region))
                        if old_best > self.threshold:
                            result_list.append(old_coords)
                            result_scores.append(old_best)

            # Deduplicate
            seen = set()
            unique_coords = []
            unique_scores = []
            for coords, score in zip(result_list, result_scores):
                key = (coords[0] // 10, coords[1] // 10)
                if key not in seen:
                    seen.add(key)
                    unique_coords.append(coords)
                    unique_scores.append(score)
            return unique_coords, unique_scores

        return kept, kept_scores

    def _match_ocr(self, frame):
        ocr = self._get_ocr()
        results, _ = ocr(frame)
        if not results:
            return None

        for (box, recognized, confidence) in results:
            if self.text.lower() in recognized.lower():
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x, y = int(min(xs)), int(min(ys))
                w, h = int(max(xs) - x), int(max(ys) - y)
                return (x, y, w, h)

        return None


def build_locator(template_path=None, text=None, coords=None, dynamic=False, detect_interval=1):
    """Build a per-frame watermark locator callable.

    Args:
        template_path: logo/template image path for template matching.
        text: watermark text for OCR detection.
        coords: static (x, y, w, h) — ignored if dynamic=True.
        dynamic: if True, re-detect every frame (template) or every N frames (OCR).
        detect_interval: for OCR mode, re-detect every N frames.

    Returns:
        callable(frame) → [(x, y, w, h), ...]
    """
    if coords is not None and not dynamic:
        # Static coords: return a trivial locator
        def _static_locator(_frame):
            return [coords]
        return _static_locator

    if template_path:
        interval = 1 if dynamic else 999999  # static: detect once
        return DynamicLocator("template", template_path=template_path, detect_interval=interval)

    if text:
        interval = detect_interval if dynamic else 999999
        return DynamicLocator("ocr", text=text, detect_interval=interval)

    raise ValueError("Must provide --coords, --logo, or --text to locate watermark")


def locate_by_text(frame, text):
    """Locate a text watermark using RapidOCR.

    Returns (x, y, w, h) or None if the text is not found.
    """
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        log.error(
            "RapidOCR is required for text-based watermark detection. "
            "Install it with: pip install rapidocr-onnxruntime"
        )
        raise

    engine = RapidOCR()
    results, _ = engine(frame)

    if not results:
        log.warning("OCR found no text in frame")
        return None

    for (box, recognized, confidence) in results:
        if text.lower() in recognized.lower():
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            log.info(
                "OCR found '%s' at (%d,%d,%d,%d) confidence=%.3f",
                recognized, x, y, w, h, confidence,
            )
            return (x, y, w, h)

    log.warning("OCR did not find text '%s' in frame", text)
    return None


def manual_select(frame, window_name="Select Watermark (drag & press ENTER, ESC to skip)"):
    """Open an OpenCV window to let the user draw a rectangle around the watermark.

    Returns (x, y, w, h) or None if the user cancels.
    """
    roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)
    x, y, w, h = roi
    if w == 0 or h == 0:
        return None
    return (int(x), int(y), int(w), int(h))


def locate_watermark(frame, template_path=None, text=None, coords=None):
    """Determine the watermark bounding box.

    Priority:
    1. Explicit *coords* (x, y, w, h) provided by the user.
    2. *template_path* – template matching.
    3. *text* – OCR-based detection.
    4. Fallback: interactive manual selection via OpenCV GUI.

    Args:
        frame: BGR numpy array (first frame of the video).
        template_path: path to a logo/template image.
        text: watermark text string.
        coords: (x, y, w, h) tuple.

    Returns:
        (x, y, w, h) tuple.
    """
    h, w = frame.shape[:2]

    if coords is not None:
        validate_coords(*coords, w, h)
        log.info("Using user-supplied coordinates: %s", coords)
        return coords

    if template_path:
        result = locate_by_template(frame, template_path)
        if result is not None:
            return result
        log.warning("Template matching failed; falling back to manual selection.")

    if text:
        result = locate_by_text(frame, text)
        if result is not None:
            return result
        log.warning("OCR detection failed; falling back to manual selection.")

    # Fallback: manual selection
    log.info("Opening interactive ROI selector window...")
    roi = manual_select(frame)
    if roi is None:
        raise RuntimeError("Watermark location not provided — cannot proceed.")
    return roi
