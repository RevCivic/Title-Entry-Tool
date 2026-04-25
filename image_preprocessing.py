"""Image pre-processing for vehicle title documents.

Provides deskew, Otsu binarization, document-border detection, and
perspective correction to maximise OCR and AI extraction accuracy for
scanned or phone-captured title images.
"""

import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_SKEW_ANGLE_DEGREES = 15.0


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV BGR numpy array."""
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to a PIL RGB image."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------


def deskew_image(image: Image.Image) -> Image.Image:
    """Correct small rotational skew caused by scanner misalignment.

    Uses Hough line detection on a binarised copy to estimate the dominant
    text angle, then rotates the image to correct it.  Returns the original
    image unchanged when the estimated angle is outside
    ±``_MAX_SKEW_ANGLE_DEGREES``, which avoids over-rotating severely tilted
    phone captures (perspective correction handles those instead).
    """
    bgr = _pil_to_bgr(image)
    gray = _to_gray(bgr)

    # Binarise for line detection.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect lines via Hough transform.
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=max(50, min(image.width, image.height) // 10),
        maxLineGap=10,
    )
    if lines is None:
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -_MAX_SKEW_ANGLE_DEGREES < angle < _MAX_SKEW_ANGLE_DEGREES:
                angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:
        return image

    h, w = bgr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        bgr, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return _bgr_to_pil(rotated)


# ---------------------------------------------------------------------------
# Otsu binarization
# ---------------------------------------------------------------------------


def binarize_image(image: Image.Image) -> Image.Image:
    """Apply adaptive Otsu binarization to improve OCR on low-contrast scans.

    Returns a grayscale PIL image with text appearing as black on white.
    """
    bgr = _pil_to_bgr(image)
    gray = _to_gray(bgr)

    # Adaptive Gaussian thresholding handles uneven lighting better than
    # global Otsu for phone-captured titles with shadows.
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(binary)


# ---------------------------------------------------------------------------
# Document border detection and crop
# ---------------------------------------------------------------------------


def _largest_quadrilateral(
    contours: list,
) -> Optional[np.ndarray]:
    """Return the largest 4-corner approximation from a list of contours."""
    best: Optional[np.ndarray] = None
    best_area = 0.0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = float(cv2.contourArea(approx))
            if area > best_area:
                best_area = area
                best = approx
    return best


def detect_and_crop_document(image: Image.Image) -> Image.Image:
    """Detect the rectangular title document within the image and crop to it.

    Useful when titles are photographed on a desk or scanner bed with a
    visible border.  Returns the original image unchanged when no clear
    rectangle is detected (e.g. the document fills the entire frame).
    """
    bgr = _pil_to_bgr(image)
    gray = _to_gray(bgr)

    # Blur to reduce noise before edge detection.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to close gaps in document borders.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Sort by area descending; only consider large contours.
    img_area = image.width * image.height
    large = [c for c in contours if cv2.contourArea(c) > img_area * 0.10]
    if not large:
        return image

    quad = _largest_quadrilateral(sorted(large, key=cv2.contourArea, reverse=True))
    if quad is None:
        return image

    x, y, w, h = cv2.boundingRect(quad)
    # Require the crop to cover at least 20% of the image to avoid tiny boxes.
    if w * h < img_area * 0.20:
        return image

    cropped = bgr[y : y + h, x : x + w]
    return _bgr_to_pil(cropped)


# ---------------------------------------------------------------------------
# Perspective correction
# ---------------------------------------------------------------------------


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Return (top-left, top-right, bottom-right, bottom-left) ordering."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_correct(image: Image.Image) -> Image.Image:
    """Apply a perspective (four-point) transform to flatten a tilted title.

    Detects the largest quadrilateral in the image and warps it to a
    top-down rectangular view.  Returns the original image when no suitable
    quadrilateral is found.
    """
    bgr = _pil_to_bgr(image)
    gray = _to_gray(bgr)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    img_area = image.width * image.height
    large = [c for c in contours if cv2.contourArea(c) > img_area * 0.15]
    if not large:
        return image

    quad = _largest_quadrilateral(sorted(large, key=cv2.contourArea, reverse=True))
    if quad is None:
        return image

    pts = quad.reshape(4, 2).astype("float32")
    rect = _order_points(pts)

    tl, tr, br, bl = rect
    width_a = float(np.linalg.norm(br - bl))
    width_b = float(np.linalg.norm(tr - tl))
    height_a = float(np.linalg.norm(tr - br))
    height_b = float(np.linalg.norm(tl - bl))
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))

    if max_width < 100 or max_height < 100:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(bgr, matrix, (max_width, max_height))
    return _bgr_to_pil(warped)


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------


def preprocess_title_image(image: Image.Image) -> Image.Image:
    """Apply the full pre-processing pipeline to a title image.

    Steps (each is a no-op when it cannot improve the image):
        1. Perspective correction  – flatten phone-captured tilted shots
        2. Document border crop    – remove scanner/desk background
        3. Deskew                  – correct small rotational misalignment

    The binarization step is intentionally *not* applied here because
    the full-colour image is needed for AI extraction.  Call
    ``binarize_image()`` separately when preparing images for Tesseract.
    """
    image = perspective_correct(image)
    image = detect_and_crop_document(image)
    image = deskew_image(image)
    return image
