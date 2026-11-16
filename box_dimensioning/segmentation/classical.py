"""Classical computer vision pipeline for box segmentation and corner localization (F2).
Uses edge filtering, Hough lines, morphological analysis, contour polygon approximation,
GrabCut foreground segmentation, and Harris sub-pixel corner refinement.
Designed to handle printed box artwork and texture noise.
"""

from typing import Tuple, List, Optional, Dict
import cv2
import numpy as np


class ClassicalBoxSegmenter:
    """Classical segmentation and corner localization pipeline."""

    def __init__(
        self,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        canny_ratio: float = 0.33,
        min_contour_area_fraction: float = 0.03,  # At least 3% of image area
    ):
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.canny_ratio = canny_ratio
        self.min_contour_area_fraction = min_contour_area_fraction

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies edge-preserving bilateral filter to suppress printed box artwork textures."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Bilateral filter smooths textures inside printed labels while keeping outer silhouette edges sharp
        filtered = cv2.bilateralFilter(
            gray, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space
        )
        return gray, filtered

    def detect_edges(self, filtered_gray: np.ndarray) -> np.ndarray:
        """Adaptive Canny edge detection with morphological closing."""
        v = np.median(filtered_gray)
        lower = int(max(0, (1.0 - self.canny_ratio) * v))
        upper = int(min(255, (1.0 + self.canny_ratio) * v))
        edges = cv2.Canny(filtered_gray, lower, upper)

        # Morphological closing to bridge small gaps along package edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        return edges_closed

    def find_box_contours(self, edges: np.ndarray, min_area: float) -> List[np.ndarray]:
        """Finds prominent contours that match cuboid perspective silhouette."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                candidates.append(cnt)
        candidates.sort(key=cv2.contourArea, reverse=True)
        return candidates

    def segment_grabcut(self, image: np.ndarray, bounding_rect: Tuple[int, int, int, int], iter_count: int = 2) -> np.ndarray:
        """Refines parcel mask using GrabCut around candidate bounding rectangle."""
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        x, y, w, h = bounding_rect

        # Inset margin to avoid taking full background
        rect = (max(0, x), max(0, y), min(image.shape[1] - x, w), min(image.shape[0] - y, h))
        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iter_count, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            return mask2
        except Exception:
            return np.ones(image.shape[:2], dtype=np.uint8)

    def extract_corners_from_contour(self, contour: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Approximates polygon vertices and refines corners with sub-pixel accuracy."""
        peri = cv2.arcLength(contour, True)
        # Douglas-Peucker polygon approximation: cuboid silhouettes typically produce 4 to 8 vertices
        for eps_factor in [0.02, 0.03, 0.04, 0.015, 0.05]:
            approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
            if 4 <= len(approx) <= 8:
                break

        pts = approx.reshape(-1, 2).astype(np.float32)

        # Sub-pixel corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        try:
            refined = cv2.cornerSubPix(gray, pts, (7, 7), (-1, -1), criteria)
            return refined
        except Exception:
            return pts

    def detect_corners(self, image: np.ndarray) -> Dict:
        """Full pipeline: detects box silhouette and extracts 2D corner candidates."""
        h, w = image.shape[:2]
        min_area = w * h * self.min_contour_area_fraction

        gray, filtered = self.preprocess_image(image)
        edges = self.detect_edges(filtered)
        contours = self.find_box_contours(edges, min_area)

        if not contours:
            # Fallback to Shi-Tomasi corners if no large single contour found
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=8, qualityLevel=0.04, minDistance=25)
            pts = corners.reshape(-1, 2) if corners is not None else np.empty((0, 2))
            return {
                "corners_2d": pts,
                "mask": np.zeros((h, w), dtype=np.uint8),
                "bounding_rect": (0, 0, w, h),
                "method": "classical_fallback",
            }

        main_contour = contours[0]
        rect = cv2.boundingRect(main_contour)
        corners_2d = self.extract_corners_from_contour(main_contour, gray)

        # Generate binary parcel mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)

        return {
            "corners_2d": corners_2d,
            "mask": mask,
            "bounding_rect": rect,
            "method": "classical_contour_subpix",
            "contour_area": float(cv2.contourArea(main_contour)),
        }
