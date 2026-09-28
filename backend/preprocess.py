import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import logging

logger = logging.getLogger(__name__)


# ─── Core preprocessing ────────────────────────────────────────────

def preprocess_image(image: Image.Image, target_size: tuple = (336, 336)) -> Image.Image:
    """
    Normalize wine label photo for better CLIP matching.
    Handles: poor lighting, angle, glare, low contrast.
    """
    img = image.convert("RGB")
    img = ImageOps.exif_transpose(img) or img
    img = _letterbox(img, target_size)
    img = _normalize_lighting(img)
    return img


def _letterbox(img: Image.Image, target_size: tuple) -> Image.Image:
    """Resize to fit within target_size, pad with neutral gray background."""
    w, h = img.size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    bg = Image.new("RGB", target_size, (128, 128, 128))
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    bg.paste(img, (offset_x, offset_y))
    return bg


def _normalize_lighting(img: Image.Image) -> Image.Image:
    """Normalize lighting, boost contrast, sharpen slightly."""
    img = ImageOps.autocontrast(img, cutoff=1)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.1)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def decode_image(data: bytes) -> Image.Image:
    """Decode image from bytes, handling various formats."""
    return Image.open(io.BytesIO(data))


# ─── Robust label extraction (Phase 1) ─────────────────────────────

def extract_label_robust(image: Image.Image):
    """
    Extract the main label region from a bottle photo using multiple strategies.

    Returns:
        (label_image, quality_score, method_name)
        quality_score: 0.0-1.0, higher = better extraction
        method_name: which strategy succeeded
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    # Try strategies in order of reliability
    strategies = [
        ("contour", _extract_by_contour),
        ("color_seg", _extract_by_color_segmentation),
        ("row_contrast", _extract_by_row_contrast),
    ]

    best_result = None
    best_score = -1.0

    for name, fn in strategies:
        try:
            crop, score = fn(rgb, h, w)
            if crop is not None and score > best_score:
                # Convert numpy crop back to PIL
                if isinstance(crop, np.ndarray):
                    crop_pil = Image.fromarray(crop)
                else:
                    crop_pil = crop
                best_result = (crop_pil, score, name)
                best_score = score
        except Exception as e:
            logger.debug(f"Label extraction strategy '{name}' failed: {e}")
            continue

    if best_result is None:
        # Absolute fallback: return the original image
        logger.warning("All label extraction strategies failed, returning original")
        return image, 0.0, "fallback_original"

    return best_result


def _extract_by_contour(rgb: np.ndarray, h: int, w: int):
    """Strategy A: Canny edge detection -> find largest rectangular contour."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection with adaptive thresholds
    median_val = np.median(blurred)
    low = int(max(0, 0.5 * median_val))
    high = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(blurred, low, high)

    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0.0

    # Filter for rectangular-ish contours that could be labels
    min_area = h * w * 0.05  # At least 5% of image
    max_area = h * w * 0.85  # At most 85% of image
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # Labels are roughly rectangular (4-8 vertices after approximation)
        if 4 <= len(approx) <= 8:
            x, y, cw, ch = cv2.boundingRect(approx)
            aspect = cw / ch if ch > 0 else 0
            # Wine labels are typically wider than tall or roughly square
            if 0.3 < aspect < 4.0 and ch > h * 0.1:
                # Score based on area and rectangularity
                rect_score = area / (cw * ch) if (cw * ch) > 0 else 0
                area_ratio = area / (h * w)
                score = 0.4 * rect_score + 0.4 * min(area_ratio * 3, 1.0) + 0.2 * (1.0 - abs(aspect - 1.0) / 3.0)
                candidates.append((x, y, cw, ch, score, approx))

    if not candidates:
        return None, 0.0

    # Sort by score, take best
    candidates.sort(key=lambda c: c[4], reverse=True)
    x, y, cw, ch, score, approx = candidates[0]

    # Add padding
    pad_x = int(cw * 0.05)
    pad_y = int(ch * 0.05)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    crop = rgb[y1:y2, x1:x2]
    return crop, min(score, 1.0)


def _extract_by_color_segmentation(rgb: np.ndarray, h: int, w: int):
    """Strategy B: Color-based segmentation — labels are usually lighter than bottle."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Wine labels are typically lighter (high value in HSV) and less saturated than the bottle
    # Threshold: high brightness, moderate-to-low saturation
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # Labels tend to be brighter than dark bottle glass
    bright_thresh = np.percentile(v_channel, 60)
    mask = (v_channel > bright_thresh).astype(np.uint8) * 255

    # Also catch white/light labels on any background
    light_mask = (v_channel > 180).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, light_mask)

    # Clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours of the bright regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0.0

    # Find the largest bright region (likely the label)
    min_area = h * w * 0.03
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None, 0.0

    x, y, cw, ch = cv2.boundingRect(best_cnt)
    area_ratio = best_area / (h * w)
    score = min(area_ratio * 2.5, 0.8)  # Cap at 0.8

    # Add padding
    pad_x = int(cw * 0.05)
    pad_y = int(ch * 0.05)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    crop = rgb[y1:y2, x1:x2]
    return crop, score


def _extract_by_row_contrast(rgb: np.ndarray, h: int, w: int):
    """Strategy C: Find highest-contrast horizontal band (original method, improved)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    row_means = gray.mean(axis=1)

    # Try multiple window sizes
    best_score = 0.0
    best_slice = (0, h)

    for window_frac in [0.25, 0.33, 0.4]:
        window = int(h * window_frac)
        for i in range(0, h - window, max(1, window // 6)):
            band = row_means[i:i + window]
            contrast = band.std()
            # Also check that the band isn't too dark (bottle) or too bright (background)
            mean_brightness = band.mean()
            if mean_brightness < 30 or mean_brightness > 240:
                continue
            score = contrast * (1.0 - abs(mean_brightness - 128) / 128)
            if score > best_score:
                best_score = score
                best_slice = (i, i + window)

    if best_score == 0:
        return None, 0.0

    y_start, y_end = best_slice
    padding = (y_end - y_start) // 8
    y_start = max(0, y_start - padding)
    y_end = min(h, y_end + padding)

    crop = rgb[y_start:y_end, :]
    normalized_score = min(best_score / 50.0, 0.6)  # Normalize to 0-0.6
    return crop, normalized_score


# ─── Glare removal (Phase 4) ──────────────────────────────────────

def remove_glare(image: Image.Image) -> Image.Image:
    """
    Detect and reduce specular highlights (glare) on glass bottles.
    Works by detecting overexposed regions and blending them with neighbors.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Detect bright spots (glare) — pixels very close to max brightness
    _, glare_mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

    # Also detect near-white regions
    _, near_white = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    glare_mask = cv2.bitwise_or(glare_mask, near_white)

    # Dilate the glare mask to cover surrounding halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare_mask = cv2.dilate(glare_mask, kernel, iterations=2)

    # If very little glare, skip
    if cv2.countNonZero(glare_mask) < rgb.shape[0] * rgb.shape[1] * 0.005:
        return image

    # Inpaint the glare regions using surrounding pixels
    result = cv2.inpaint(rgb, glare_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(result)


# ─── Background suppression (Phase 4) ─────────────────────────────

def suppress_background(image: Image.Image, label_region=None) -> Image.Image:
    """
    Darken/blur regions outside the detected label area.
    If label_region is None, attempts to detect it automatically.
    """
    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]

    if label_region is not None:
        # Create mask from provided label region coordinates
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = label_region
        mask[y1:y2, x1:x2] = 255
    else:
        # Auto-detect label region using color segmentation
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        bright_thresh = np.percentile(v_channel, 55)
        mask = (v_channel > bright_thresh).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Blur the mask edges for smooth transition
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)

    # Apply: keep original where mask is white, darken where mask is black
    mask_3ch = mask_blurred[:, :, np.newaxis].astype(np.float32) / 255.0
    darkened = (rgb * 0.2).astype(np.float32)  # Very dark background
    result = (rgb.astype(np.float32) * mask_3ch + darkened * (1.0 - mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)


# ─── OCR preprocessing (Phase 4) ──────────────────────────────────

def normalize_for_ocr(image: Image.Image) -> Image.Image:
    """
    Improved OCR preprocessing: adaptive thresholding, morphological ops,
    contrast enhancement optimized for text detection.
    """
    img = image.convert("RGB")
    img = ImageOps.exif_transpose(img) or img

    # Remove glare first
    img = remove_glare(img)

    # Auto-contrast
    img = ImageOps.autocontrast(img, cutoff=2)

    # Convert to grayscale
    gray = np.array(img.convert("L"))

    # Adaptive thresholding (better than global for uneven lighting)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    # Morphological cleanup: remove noise, connect broken characters
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    result = Image.fromarray(binary)

    # Scale up for better OCR accuracy
    w, h = result.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        result = result.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return result


# ─── Backward-compatible wrapper ───────────────────────────────────

def extract_label_region(image: Image.Image) -> Image.Image:
    """
    Extract label region — now uses robust multi-strategy extraction.
    Backward-compatible: returns just the PIL Image.
    """
    label_img, score, method = extract_label_robust(image)
    return label_img
