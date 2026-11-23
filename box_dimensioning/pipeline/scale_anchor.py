"""Scale Anchor Module (Step 4 Upgrade) — Resolves Monocular Scale Ambiguity.

Problem: Structure-from-Motion (SfM) gives only metric-free 3D (scale unknown).
Solution: User places a known reference object next to the box. We detect it and
compute absolute scale from the reference object's known real-world size.

Supported Reference Objects:
1. A4 paper     (297 × 210 mm)
2. Credit card  (85.6 × 54 mm) — ISO/IEC 7810 ID-1
3. A5 paper     (210 × 148 mm)
4. Letter paper (279 × 216 mm)
5. Business card (90 × 55 mm)

Workflow:
  - User optionally selects reference type in the UI
  - detect_reference() finds the reference rectangle in the image
  - compute_scale_factor() returns mm-per-pixel ratio
  - scale_dimensions() applies ratio to raw 3D measurements
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Reference object registry (name → real dimensions in mm)
# ---------------------------------------------------------------------------
REFERENCE_OBJECTS: Dict[str, Tuple[float, float]] = {
    "A4 Paper":      (297.0, 210.0),
    "A5 Paper":      (210.0, 148.0),
    "Letter Paper":  (279.0, 216.0),
    "Credit Card":   (85.6,  54.0),
    "Business Card": (90.0,  55.0),
}

DEFAULT_REFERENCE = "A4 Paper"


def detect_reference_rectangle(
    image: np.ndarray,
    ref_name: str = DEFAULT_REFERENCE,
    expected_ratio_tolerance: float = 0.15,
) -> Optional[Dict]:
    """Detects the reference object rectangle in the image using contour + aspect ratio.

    Strategy:
    1. Find all large rectangular contours (4–6 vertices).
    2. Filter by expected aspect ratio of the reference object (± tolerance).
    3. Pick the one with the best matching aspect ratio.

    Returns dict with:
        - 'corners': (4, 2) array of corner points in image pixels
        - 'width_px': pixel width of reference
        - 'height_px': pixel height of reference
        - 'scale_mm_per_px': real-world mm per pixel
        - 'ref_name': which reference was matched
    """
    if ref_name not in REFERENCE_OBJECTS:
        ref_name = DEFAULT_REFERENCE

    real_long, real_short = REFERENCE_OBJECTS[ref_name]
    expected_ratio = real_long / real_short  # always >= 1.0

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # --- Edge detection ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_match = None
    best_ratio_error = float('inf')
    min_area = w * h * 0.005  # Reference must be at least 0.5% of image

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        for eps in [0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4:
                break

        if len(approx) != 4:
            continue

        pts = approx.reshape(4, 2).astype(np.float32)
        # Order corners: top-left, top-right, bottom-right, bottom-left
        pts = _order_corners(pts)

        # Compute width and height using perspective-aware measurement
        width_px  = float(np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3])) / 2.0
        height_px = float(np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1])) / 2.0

        if width_px < 10 or height_px < 10:
            continue

        obs_ratio = max(width_px, height_px) / min(width_px, height_px)
        ratio_error = abs(obs_ratio - expected_ratio) / expected_ratio

        if ratio_error < expected_ratio_tolerance and ratio_error < best_ratio_error:
            best_ratio_error = ratio_error
            # Scale: pixels → mm
            scale_long = real_long / max(width_px, height_px)
            scale_short = real_short / min(width_px, height_px)
            scale_mm_per_px = (scale_long + scale_short) / 2.0

            best_match = {
                "corners": pts,
                "width_px": float(width_px),
                "height_px": float(height_px),
                "scale_mm_per_px": float(scale_mm_per_px),
                "ref_name": ref_name,
                "ratio_error": float(ratio_error),
                "contour_area_px2": float(area),
            }

    return best_match


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Orders 4 corner points as: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_absolute_scale(
    image: np.ndarray,
    ref_name: str = DEFAULT_REFERENCE,
) -> Optional[float]:
    """Returns mm-per-pixel scale factor if reference is detected, else None."""
    result = detect_reference_rectangle(image, ref_name)
    if result is None:
        return None
    return result["scale_mm_per_px"]


def scale_dimensions_to_cm(
    raw_dims: Tuple[float, float, float],
    scale_mm_per_px: Optional[float] = None,
    pixel_dims: Optional[Tuple[float, float, float]] = None,
    fallback_distance_m: float = 0.55,
    focal_length_px: float = 1000.0,
    raw_unit: str = "cm",
) -> Tuple[float, float, float]:
    """
    Converts raw 3D measurement units to real-world cm using scale anchor.

    If scale_mm_per_px is provided (reference object detected):
        → If pixel_dims provided: dim_cm = px * scale_mm_per_px / 10
        → Else if raw_unit == "px": dim_cm = raw_px * scale_mm_per_px / 10
        → Else: preserves raw cm or scales from meters
    Else (no reference):
        → If raw_unit == "cm": dimensions are already in cm (no extra scaling)
        → If raw_unit == "m": converts meters to cm (raw_m * 100)
        → If raw_unit == "px": uses pinhole monocular fallback: (px * dist / f) * 100

    Args:
        raw_dims: (L, W, H) in raw_unit
        scale_mm_per_px: mm per pixel from reference object (or None)
        pixel_dims: optional (W_px, H_px, D_px) in pixels
        fallback_distance_m: assumed scanning distance if no reference (~0.55m)
        focal_length_px: camera focal length in pixels (for fallback)
        raw_unit: unit of raw_dims ("cm", "m", or "px")

    Returns:
        (length_cm, width_cm, height_cm) sorted (L >= W >= H) and clamped to realistic courier parcel bounds.
    """
    if scale_mm_per_px is not None and scale_mm_per_px > 0:
        if pixel_dims is not None and len(pixel_dims) >= 2:
            l_px = float(pixel_dims[0])
            w_px = float(pixel_dims[1])
            h_px = float(pixel_dims[2]) if len(pixel_dims) > 2 else min(l_px, w_px) * 0.65
            length_cm = float(l_px * scale_mm_per_px / 10.0)
            width_cm  = float(w_px * scale_mm_per_px / 10.0)
            height_cm = float(h_px * scale_mm_per_px / 10.0)
        elif raw_unit == "px":
            l_raw, w_raw, h_raw = raw_dims
            length_cm = float(l_raw * scale_mm_per_px / 10.0)
            width_cm  = float(w_raw * scale_mm_per_px / 10.0)
            height_cm = float(h_raw * scale_mm_per_px / 10.0)
        elif raw_unit == "m":
            length_cm = float(raw_dims[0] * 100.0)
            width_cm  = float(raw_dims[1] * 100.0)
            height_cm = float(raw_dims[2] * 100.0)
        else:
            length_cm = float(raw_dims[0])
            width_cm  = float(raw_dims[1])
            height_cm = float(raw_dims[2])
    else:
        if raw_unit == "cm":
            length_cm = float(raw_dims[0])
            width_cm  = float(raw_dims[1])
            height_cm = float(raw_dims[2])
        elif raw_unit == "m":
            length_cm = float(raw_dims[0] * 100.0)
            width_cm  = float(raw_dims[1] * 100.0)
            height_cm = float(raw_dims[2] * 100.0)
        elif raw_unit == "px":
            l_raw, w_raw, h_raw = raw_dims
            f_safe = max(100.0, float(focal_length_px))
            length_cm = float((l_raw * fallback_distance_m / f_safe) * 100.0)
            width_cm  = float((w_raw * fallback_distance_m / f_safe) * 100.0)
            height_cm = float((h_raw * fallback_distance_m / f_safe) * 100.0)
        else:
            length_cm = float(raw_dims[0])
            width_cm  = float(raw_dims[1])
            height_cm = float(raw_dims[2])

    # Sort so length >= width >= height
    sorted_dims = sorted([length_cm, width_cm, height_cm], reverse=True)
    length_cm, width_cm, height_cm = sorted_dims[0], sorted_dims[1], sorted_dims[2]

    # Clamp to realistic courier parcel bounds
    length_cm = float(np.clip(length_cm, 5.0, 120.0))
    width_cm  = float(np.clip(width_cm,  4.0, 100.0))
    height_cm = float(np.clip(height_cm, 2.0, 80.0))

    return length_cm, width_cm, height_cm


def draw_reference_overlay(
    image: np.ndarray,
    detection: Dict,
) -> np.ndarray:
    """Draws detected reference object overlay with scale info on image."""
    img = image.copy()
    pts = detection["corners"].astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=3)

    ref_name = detection.get("ref_name", "Reference")
    scale = detection.get("scale_mm_per_px", 0.0)
    label = f"{ref_name} | {scale:.3f} mm/px"

    # Label background
    x, y = pts[0]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x - 2, y - th - 10), (x + tw + 4, y + 2), (0, 255, 255), -1)
    cv2.putText(img, label, (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img
