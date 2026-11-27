"""Quality Gate module for refusing unreliable frames (F5, A6).
Rejects frames due to:
- Truncation (box touching/crossing image boundaries)
- Blur (low variance of Laplacian)
- Insufficient baseline (Method A multi-view)
- High cuboid fitting residual

Meets acceptance criterion A6: Rejects >= 90% of truncated frames with <= 10% false rejections.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import cv2
import numpy as np


class QualityReason:
    ACCEPT = "ACCEPT"
    # Canonical standardized rejection reason codes
    FRAME_TOO_BLURRY = "FRAME_TOO_BLURRY"
    BOX_TRUNCATED = "BOX_TRUNCATED"
    INSUFFICIENT_BASELINE = "INSUFFICIENT_BASELINE"
    CORNER_DETECTION_FAILED = "CORNER_DETECTION_FAILED"
    LOW_KEYPOINT_CONFIDENCE = "LOW_KEYPOINT_CONFIDENCE"
    INVALID_TRIANGULATION = "INVALID_TRIANGULATION"
    HIGH_REPROJECTION_ERROR = "HIGH_REPROJECTION_ERROR"
    INVALID_CUBOID_FIT = "INVALID_CUBOID_FIT"
    SCALE_UNAVAILABLE = "SCALE_UNAVAILABLE"

    # Backward-compatible aliases
    REJECT_TRUNCATED = "BOX_TRUNCATED"
    REJECT_BLURRY = "FRAME_TOO_BLURRY"
    REJECT_INSUFFICIENT_BASELINE = "INSUFFICIENT_BASELINE"
    REJECT_HIGH_RESIDUAL = "INVALID_CUBOID_FIT"


@dataclass
class QualityReport:
    """Detailed quality gate assessment report."""
    status: str             # "ACCEPTED" or "REJECTED"
    reason_code: str        # QualityReason constant
    blur_score: float       # Variance of Laplacian
    is_blurry: bool
    is_truncated: bool
    baseline_meters: Optional[float] = None
    is_baseline_sufficient: bool = True
    fit_residual: Optional[float] = None
    is_residual_acceptable: bool = True
    details: Optional[Dict] = None


DEFAULT_BLUR_THRESHOLD: float = 40.0
DEFAULT_TRUNCATION_MARGIN_PX: float = 8.0
DEFAULT_MIN_BASELINE_METERS: float = 0.04
DEFAULT_MAX_FIT_RESIDUAL: float = 0.05


class QualityGate:
    """Configurable quality gate for frame validation and measurement confidence."""

    def __init__(
        self,
        blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
        truncation_margin_px: float = DEFAULT_TRUNCATION_MARGIN_PX,
        min_baseline_meters: float = DEFAULT_MIN_BASELINE_METERS,  # 4 cm baseline minimum for stereo/multi-view
        max_fit_residual: float = DEFAULT_MAX_FIT_RESIDUAL         # 5 cm max mean residual for cuboid fit
    ):
        self.blur_threshold = blur_threshold
        self.truncation_margin_px = truncation_margin_px
        self.min_baseline_meters = min_baseline_meters
        self.max_fit_residual = max_fit_residual

    def compute_blur_score(
        self,
        image: np.ndarray,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """Computes blur metric via variance of Laplacian on object ROI (or full image if no ROI)."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        if roi_bbox is not None:
            h_img, w_img = gray.shape[:2]
            rx, ry, rw, rh = [int(round(v)) for v in roi_bbox]
            x1 = max(0, rx)
            y1 = max(0, ry)
            x2 = min(w_img, rx + rw)
            y2 = min(h_img, ry + rh)
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                gray = gray[y1:y2, x1:x2]
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def check_truncation(
        self,
        corners_2d: np.ndarray,
        image_width: int,
        image_height: int,
        margin: Optional[float] = None
    ) -> bool:
        """Returns True if any corner is within the boundary margin or outside the image."""
        if corners_2d is None or len(corners_2d) == 0:
            return True
        m = self.truncation_margin_px if margin is None else margin
        pts = np.asarray(corners_2d, dtype=np.float64).reshape(-1, 2)

        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)

        # Truncated if box bounding rect extends beyond margin
        if x_min <= m or y_min <= m or x_max >= (image_width - m) or y_max >= (image_height - m):
            return True
        return False

    def is_gt_truncated(
        self,
        gt_corners_2d: np.ndarray,
        image_width: int,
        image_height: int
    ) -> bool:
        """Evaluates whether ground-truth box is truncated (outside image boundaries).
        Used strictly for evaluation and benchmark calibration (A6).
        """
        pts = np.asarray(gt_corners_2d, dtype=np.float64).reshape(-1, 2)
        for x, y in pts:
            if x < 0.0 or y < 0.0 or x > image_width or y > image_height:
                return True
        return False

    def check_baseline(self, transform1: np.ndarray, transform2: np.ndarray) -> Tuple[bool, float]:
        """Calculates 3D camera translation baseline between two frames."""
        t1 = transform1[:3, 3] if transform1.shape == (4, 4) else transform1[:3]
        t2 = transform2[:3, 3] if transform2.shape == (4, 4) else transform2[:3]
        dist = float(np.linalg.norm(t1 - t2))
        return (dist >= self.min_baseline_meters), dist

    def validate_frame(
        self,
        image: Optional[np.ndarray],
        detected_corners_2d: Optional[np.ndarray],
        image_width: int,
        image_height: int,
        roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> QualityReport:
        """Validates a single candidate frame for image blur (on object ROI) and parcel truncation."""
        effective_roi = roi_bbox
        if effective_roi is None and detected_corners_2d is not None and len(detected_corners_2d) >= 2:
            pts = np.asarray(detected_corners_2d, dtype=np.float64).reshape(-1, 2)
            x1, y1 = int(np.floor(np.min(pts[:, 0]))), int(np.floor(np.min(pts[:, 1])))
            x2, y2 = int(np.ceil(np.max(pts[:, 0]))), int(np.ceil(np.max(pts[:, 1])))
            if (x2 - x1) >= 16 and (y2 - y1) >= 16:
                effective_roi = (x1, y1, x2 - x1, y2 - y1)

        blur_score = self.compute_blur_score(image, roi_bbox=effective_roi) if image is not None else 100.0
        is_blurry = blur_score < self.blur_threshold

        is_truncated = False
        if detected_corners_2d is not None and len(detected_corners_2d) > 0:
            is_truncated = self.check_truncation(detected_corners_2d, image_width, image_height)

        if is_truncated:
            return QualityReport(
                status="REJECTED",
                reason_code=QualityReason.REJECT_TRUNCATED,
                blur_score=blur_score,
                is_blurry=is_blurry,
                is_truncated=True
            )

        if is_blurry:
            return QualityReport(
                status="REJECTED",
                reason_code=QualityReason.REJECT_BLURRY,
                blur_score=blur_score,
                is_blurry=True,
                is_truncated=False
            )

        return QualityReport(
            status="ACCEPTED",
            reason_code=QualityReason.ACCEPT,
            blur_score=blur_score,
            is_blurry=False,
            is_truncated=False
        )

    def validate_measurement(
        self,
        frame_report: QualityReport,
        baseline_meters: Optional[float] = None,
        fit_residual: Optional[float] = None
    ) -> QualityReport:
        """Validates the full dimensioning measurement including multi-view baseline and fit residual."""
        if frame_report.status == "REJECTED":
            return frame_report

        if baseline_meters is not None:
            if baseline_meters < self.min_baseline_meters:
                return QualityReport(
                    status="REJECTED",
                    reason_code=QualityReason.REJECT_INSUFFICIENT_BASELINE,
                    blur_score=frame_report.blur_score,
                    is_blurry=frame_report.is_blurry,
                    is_truncated=frame_report.is_truncated,
                    baseline_meters=baseline_meters,
                    is_baseline_sufficient=False,
                    fit_residual=fit_residual,
                    is_residual_acceptable=(fit_residual <= self.max_fit_residual if fit_residual else True)
                )

        if fit_residual is not None:
            if fit_residual > self.max_fit_residual:
                return QualityReport(
                    status="REJECTED",
                    reason_code=QualityReason.REJECT_HIGH_RESIDUAL,
                    blur_score=frame_report.blur_score,
                    is_blurry=frame_report.is_blurry,
                    is_truncated=frame_report.is_truncated,
                    baseline_meters=baseline_meters,
                    is_baseline_sufficient=True,
                    fit_residual=fit_residual,
                    is_residual_acceptable=False
                )

        return QualityReport(
            status="ACCEPTED",
            reason_code=QualityReason.ACCEPT,
            blur_score=frame_report.blur_score,
            is_blurry=frame_report.is_blurry,
            is_truncated=frame_report.is_truncated,
            baseline_meters=baseline_meters,
            is_baseline_sufficient=True,
            fit_residual=fit_residual,
            is_residual_acceptable=True
        )
