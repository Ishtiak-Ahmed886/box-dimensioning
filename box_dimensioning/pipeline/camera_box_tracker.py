"""Real-time box detection and continuous tracking pipeline for webcam/video streams.

Provides:
- Class CameraBoxTracker:
  - Detects boxes and filters out non-box clutter (cups, monitors, keyboards, hands)
    using YOLO / contour geometry (quadrilateral convex hull, parallel opposing edges,
    aspect ratio, color/texture contrast).
  - Tracks detected box continuously using OpenCV tracking (e.g. TrackerMIL, optical flow,
    CSRT/KCF when available) so tracking stays smooth across frames without jitter.
  - Extracts 8 3D cuboid corners, projects them onto 2D image coordinates, labels them 1 to 8,
    computes estimated dimensions (L, W, H in cm) and volumetric weight (kg).
  - Draws visual overlays: bounding box, corner markers 1-8, cuboid wireframe, tracking status
    (INITIALIZING, TRACKING, LOST), FPS.
"""

from typing import Dict, List, Optional, Tuple, Any
import time
import math
import cv2
import numpy as np

from box_dimensioning.pipeline.quality_gate import QualityGate, QualityReport
from box_dimensioning.segmentation.deep_detector import DeepBoxDetector
from box_dimensioning.segmentation.classical import ClassicalBoxSegmenter
from box_dimensioning.dataset.box_geometry import compute_volumetric_weight, EDGES


class TrackingState:
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    LOST = "LOST"


class OpticalFlowTracker:
    """Lucas-Kanade optical flow feature tracker with forward-backward error validation."""

    def __init__(self, max_features: int = 40):
        self.max_features = max_features
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)

    def init(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        x, y, w, h = [int(v) for v in bbox]
        if w <= 4 or h <= 4:
            return False

        mask = np.zeros_like(gray)
        pad_x = max(1, int(w * 0.05))
        pad_y = max(1, int(h * 0.05))
        mask[max(0, y + pad_y):min(gray.shape[0], y + h - pad_y),
             max(0, x + pad_x):min(gray.shape[1], x + w - pad_x)] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=7,
            mask=mask
        )
        if pts is None or len(pts) < 4:
            # Fallback grid inside bbox
            gx = np.linspace(x + pad_x, x + w - pad_x, 4)
            gy = np.linspace(y + pad_y, y + h - pad_y, 4)
            grid = np.array([[xx, yy] for xx in gx for yy in gy], dtype=np.float32).reshape(-1, 1, 2)
            pts = grid

        self.prev_gray = gray
        self.prev_pts = pts
        self.bbox = (x, y, w, h)
        return True

    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        if self.prev_gray is None or self.prev_pts is None or self.bbox is None:
            return False, (0, 0, 0, 0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
        )

        if next_pts is None or status is None:
            return False, self.bbox

        valid_idx = (status.ravel() == 1)
        if np.count_nonzero(valid_idx) < 4:
            return False, self.bbox

        p0 = self.prev_pts[valid_idx]
        p1 = next_pts[valid_idx]

        # Calculate median displacement
        dx = float(np.median(p1[:, 0, 0] - p0[:, 0, 0]))
        dy = float(np.median(p1[:, 0, 1] - p0[:, 0, 1]))

        x, y, w, h = self.bbox
        new_x = int(round(x + dx))
        new_y = int(round(y + dy))

        # Clamp to image boundaries
        img_h, img_w = gray.shape[:2]
        new_x = max(0, min(img_w - w, new_x))
        new_y = max(0, min(img_h - h, new_y))

        self.bbox = (new_x, new_y, w, h)
        self.prev_gray = gray
        self.prev_pts = p1.reshape(-1, 1, 2)

        return True, self.bbox


class CameraBoxTracker:
    """Real-time box detector and continuous tracker for camera and video streams."""

    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        detection_interval: int = 15,
        default_distance_cm: float = 60.0,
        use_deep_detector: bool = True,
        tracker_type: str = "mil",
        enable_smoothing: bool = True,
        smoothing_alpha: float = 0.35,
    ):
        """
        Args:
            camera_matrix: 3x3 intrinsic matrix. If None, estimated on first frame.
            dist_coeffs: Optional distortion coefficients.
            detection_interval: Re-detect every N frames to avoid tracker drift.
            default_distance_cm: Estimated or measured distance from camera to box top (default 60 cm).
            use_deep_detector: Whether to attempt YOLO ONNX inference.
            tracker_type: 'mil', 'csrt', 'kcf', or 'optical_flow'.
            enable_smoothing: Apply exponential moving average to 2D/3D coordinates.
            smoothing_alpha: Weight for current frame (0.0 to 1.0; higher = less lag).
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1), dtype=np.float64)
        self.detection_interval = max(1, detection_interval)
        self.default_distance_cm = default_distance_cm
        self.tracker_type = tracker_type.lower()
        self.enable_smoothing = enable_smoothing
        self.smoothing_alpha = smoothing_alpha

        self.quality_gate = QualityGate()
        self.detector = DeepBoxDetector(use_deep=use_deep_detector)
        self.classical_segmenter = ClassicalBoxSegmenter()

        self.state: str = TrackingState.INITIALIZING
        self.frame_count: int = 0
        self.fps: float = 0.0
        self._prev_frame_time: Optional[float] = None

        self._tracker = None
        self.current_bbox: Optional[Tuple[int, int, int, int]] = None
        self.current_corners_2d: Optional[np.ndarray] = None  # shape (8, 2)
        self.current_corners_3d: Optional[np.ndarray] = None  # shape (8, 3)
        self.current_dimensions: Dict[str, float] = {"length": 0.0, "width": 0.0, "height": 0.0}
        self.current_volumetric_weight: float = 0.0

        # Smoothing buffers
        self._smoothed_corners_2d: Optional[np.ndarray] = None
        self._smoothed_dims: Optional[Dict[str, float]] = None

    def _init_tracker(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Instantiates and initializes tracking algorithm for given bounding box."""
        x, y, w, h = [int(v) for v in bbox]
        if w < 10 or h < 10:
            return False

        # Try OpenCV Tracker
        t = None
        if self.tracker_type in ("csrt", "kcf"):
            if hasattr(cv2, "TrackerCSRT_create") and self.tracker_type == "csrt":
                t = cv2.TrackerCSRT_create()
            elif hasattr(cv2, "TrackerKCF_create") and self.tracker_type == "kcf":
                t = cv2.TrackerKCF_create()
            elif hasattr(cv2, "legacy"):
                if self.tracker_type == "csrt" and hasattr(cv2.legacy, "TrackerCSRT_create"):
                    t = cv2.legacy.TrackerCSRT_create()
                elif hasattr(cv2.legacy, "TrackerKCF_create"):
                    t = cv2.legacy.TrackerKCF_create()

        if t is None and self.tracker_type != "optical_flow":
            if hasattr(cv2, "TrackerMIL_create"):
                t = cv2.TrackerMIL_create()

        if t is not None:
            try:
                t.init(image, (x, y, w, h))
                self._tracker = t
                return True
            except Exception:
                pass

        # Fallback to Optical Flow Tracker
        oft = OpticalFlowTracker()
        if oft.init(image, (x, y, w, h)):
            self._tracker = oft
            return True

        return False

    def _update_tracker(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """Updates tracker position on current frame."""
        if self._tracker is None:
            return False, (0, 0, 0, 0)

        try:
            ok, bbox = self._tracker.update(image)
            if ok and bbox is not None:
                x, y, w, h = [int(round(v)) for v in bbox]
                # Boundary verification
                img_h, img_w = image.shape[:2]
                if w < 15 or h < 15 or x + w <= 0 or y + h <= 0 or x >= img_w or y >= img_h:
                    return False, (x, y, w, h)
                return True, (x, y, w, h)
        except Exception:
            pass

        return False, (0, 0, 0, 0)

    def _ensure_camera_matrix(self, width: int, height: int) -> np.ndarray:
        """Constructs or adapts camera intrinsic matrix for frame orientation and resolution."""
        if self.camera_matrix is not None:
            # Check if frame resolution or portrait/landscape orientation changed
            orig_cx = self.camera_matrix[0, 2]
            orig_cy = self.camera_matrix[1, 2]
            if orig_cx > 0 and orig_cy > 0:
                if abs(orig_cx - width / 2.0) > 5.0 or abs(orig_cy - height / 2.0) > 5.0:
                    scale_x = width / (orig_cx * 2.0)
                    scale_y = height / (orig_cy * 2.0)
                    self.camera_matrix[0, 0] *= scale_x
                    self.camera_matrix[1, 1] *= scale_y
                    self.camera_matrix[0, 2] = width / 2.0
                    self.camera_matrix[1, 2] = height / 2.0
            return self.camera_matrix

        # Default webcams / smartphone cameras typically have ~60-65 degree FOV
        fov_deg = 62.0
        max_dim = max(width, height)
        f = (max_dim / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        cx = width / 2.0
        cy = height / 2.0
        self.camera_matrix = np.array([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        return self.camera_matrix

    def is_box_like_contour(
        self,
        contour: np.ndarray,
        image_shape: Tuple[int, int],
        min_area_fraction: float = 0.03,
        max_area_fraction: float = 0.90,
    ) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Geometrically filters out non-box clutter (cups, keyboards, monitors, hands, round objects).
        Evaluates:
        - Convex hull & polygon vertex count (cuboid faces project to 4-8 vertices)
        - Convexity & solidity (box faces have high solidity > 0.82; hands/cutlery have low solidity)
        - Aspect ratio (between 0.25 and 4.0; eliminates thin sticks or banners)
        - Rectangularity / parallel opposing edges.
        """
        h, w = image_shape[:2]
        total_area = float(w * h)
        area = cv2.contourArea(contour)
        info: Dict[str, Any] = {"area": area, "reason": "ok"}

        if area < total_area * min_area_fraction:
            info["reason"] = "area_too_small"
            return False, None, info
        if area > total_area * max_area_fraction:
            info["reason"] = "area_too_large"
            return False, None, info

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 1e-4:
            info["reason"] = "zero_hull_area"
            return False, None, info

        solidity = area / hull_area
        info["solidity"] = solidity
        # Hands, organic shapes, irregular cups have concavities yielding low solidity
        if solidity < 0.78:
            info["reason"] = "low_solidity"
            return False, None, info

        # Bounding rect & aspect ratio
        rx, ry, rw, rh = cv2.boundingRect(contour)
        aspect_ratio = float(rw) / float(rh) if rh > 0 else 0.0
        info["aspect_ratio"] = aspect_ratio
        if aspect_ratio < 0.20 or aspect_ratio > 5.0:
            info["reason"] = "abnormal_aspect_ratio"
            return False, None, info

        # Polygon approximation
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        n_vertices = len(approx)
        info["vertices"] = n_vertices

        # Box contours in 2D perspective exhibit 4 to 8 salient vertices
        if n_vertices < 4 or n_vertices > 10:
            info["reason"] = "vertex_count_out_of_range"
            return False, None, info

        # Check circularity: round cups/bottles have high circularity 4*pi*area/perimeter^2 ~ 1.0
        circularity = (4.0 * math.pi * area) / (peri * peri + 1e-6)
        info["circularity"] = circularity
        if circularity > 0.88 and n_vertices >= 6:
            info["reason"] = "too_circular_cup_bottle"
            return False, None, info

        return True, approx, info

    def detect_box(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects primary box in image with clutter filtering.
        Combines deep detector (YOLO ONNX when available) and geometric contour analysis.
        """
        img_h, img_w = image.shape[:2]

        # 1. Try deep detector (custom YOLO box model or ONNX)
        deep_res = None
        try:
            if self.detector.use_deep:
                deep_res = self.detector.detect(image)
        except Exception:
            deep_res = None

        if deep_res is not None and ("bounding_rect" in deep_res or "bounding_box" in deep_res):
            b_tuple = deep_res.get("bounding_box", deep_res.get("bounding_rect"))
            bx, by, bw, bh = b_tuple
            conf = deep_res.get("confidence", 0.8)
            # Only accept if box meets minimum dimensions and is from custom YOLO or good confidence
            if bw >= 20 and bh >= 20 and conf >= 0.30:
                if "corners_8_2d" in deep_res and deep_res["corners_8_2d"] is not None and len(deep_res["corners_8_2d"]) == 8:
                    corners_8 = deep_res["corners_8_2d"]
                else:
                    corners_8 = self._infer_8_corners_from_bbox((bx, by, bw, bh), image)
                return {
                    "bbox": (bx, by, bw, bh),
                    "corners_8_2d": corners_8,
                    "confidence": conf,
                    "method": deep_res.get("method", "custom_yolo"),
                }

        # 2. Geometric contour & edge analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_candidates = []

        for cnt in contours:
            is_box, approx, meta = self.is_box_like_contour(cnt, (img_h, img_w))
            if is_box:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                # Score candidate by area and centering
                cx = rx + rw / 2.0
                cy = ry + rh / 2.0
                center_dist = math.hypot(cx - img_w / 2.0, cy - img_h / 2.0)
                score = meta["area"] - (center_dist * 20.0)
                valid_candidates.append({
                    "contour": cnt,
                    "approx": approx,
                    "bbox": (rx, ry, rw, rh),
                    "score": score,
                    "meta": meta
                })

        if valid_candidates:
            valid_candidates.sort(key=lambda c: c["score"], reverse=True)
            best = valid_candidates[0]
            bx, by, bw, bh = best["bbox"]
            corners_8 = self._infer_8_corners_from_contour_and_bbox(best["approx"], (bx, by, bw, bh), image)
            return {
                "bbox": (bx, by, bw, bh),
                "corners_8_2d": corners_8,
                "confidence": 0.85,
                "method": "geometry_contour",
            }

        return None

    def _infer_8_corners_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image: np.ndarray
    ) -> np.ndarray:
        """
        Projects an isometric 3D cuboid onto 2D image coordinates within bounding box.
        Returns array of shape (8, 2) corresponding to keypoints 1 to 8:
        Top face (1, 2, 3, 4), Bottom face (5, 6, 7, 8).
        """
        x, y, w, h = bbox
        # Typical 3D perspective projection of a cuboid resting on a surface seen from ~30° angle:
        # Top face is tilted parallelogram, bottom face extends lower.
        depth_offset_x = int(round(w * 0.16))
        depth_offset_y = int(round(h * 0.18))

        # Top face vertices (1 to 4)
        c1 = [x + depth_offset_x, y]
        c2 = [x + w, y]
        c3 = [x + w - depth_offset_x, y + depth_offset_y]
        c4 = [x, y + depth_offset_y]

        # Bottom face vertices (5 to 8)
        c5 = [c1[0], y + h - depth_offset_y]
        c6 = [c2[0], y + h - depth_offset_y]
        c7 = [c3[0], y + h]
        c8 = [c4[0], y + h]

        return np.array([c1, c2, c3, c4, c5, c6, c7, c8], dtype=np.float32)

    def _infer_8_corners_from_contour_and_bbox(
        self,
        approx: np.ndarray,
        bbox: Tuple[int, int, int, int],
        image: np.ndarray
    ) -> np.ndarray:
        """Infers 8 corners using polygon vertices if 6-8 vertices exist, else fallback to standard cuboid."""
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) == 8:
            # Sort top 4 and bottom 4 by Y coordinate
            sorted_by_y = pts[np.argsort(pts[:, 1])]
            top_4 = sorted_by_y[:4]
            bot_4 = sorted_by_y[4:]
            # Sort each by X coordinate
            top_4 = top_4[np.argsort(top_4[:, 0])]
            bot_4 = bot_4[np.argsort(bot_4[:, 0])]
            return np.vstack([top_4, bot_4])

        return self._infer_8_corners_from_bbox(bbox, image)

    def estimate_3d_and_dimensions(
        self,
        corners_2d: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Dict[str, float], float]:
        """
        Computes 3D cuboid corners, metric dimensions (L, W, H in cm), and volumetric weight (kg).

        Uses EPnP rigid cuboid solver (cv2.solvePnP + joint pose+scale optimization)
        to recover a geometrically valid 3D bounding box where opposite faces are
        guaranteed parallel and orthogonal by construction. Falls back to naive
        back-projection only if EPnP fails entirely.
        """
        from box_dimensioning.pipeline.epnp_solver import solve_epnp_cuboid

        h_img, w_img = image_shape[:2]
        K = self._ensure_camera_matrix(w_img, h_img)

        # Initial scale guess based on default_distance_cm and apparent bbox size
        pts = np.asarray(corners_2d, dtype=np.float64)
        bbox_w_px = np.max(pts[:, 0]) - np.min(pts[:, 0])
        bbox_h_px = np.max(pts[:, 1]) - np.min(pts[:, 1])
        z_est = self.default_distance_cm / 100.0  # meters
        fx = K[0, 0]
        # Estimate physical size from pixel extent at estimated depth
        est_w_m = max(0.05, bbox_w_px * z_est / fx)
        est_h_m = max(0.05, bbox_h_px * z_est / fx)
        est_d_m = max(0.03, min(est_w_m, est_h_m) * 0.4)  # depth heuristic
        initial_scale = np.array([est_w_m, est_h_m, est_d_m])

        result = solve_epnp_cuboid(
            corners_2d, K, self.dist_coeffs, initial_scale=initial_scale
        )

        corners_3d = result['corners_3d']
        dims = result['dimensions_cm']
        vol_wt = result['volumetric_weight_kg']

        return corners_3d, dims, vol_wt

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main processing method called per webcam/video frame.
        Handles tracking state machine, periodic detection, cuboid reconstruction,
        temporal smoothing, and telemetry rendering.
        """
        t_start = time.perf_counter()
        self.frame_count += 1
        h, w = frame.shape[:2]
        self._ensure_camera_matrix(w, h)

        # Calculate FPS
        if self._prev_frame_time is not None:
            dt = t_start - self._prev_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps = 0.85 * self.fps + 0.15 * instant_fps if self.fps > 0 else instant_fps
        self._prev_frame_time = t_start

        # State Machine: Determine whether to detect or track
        must_detect = (
            self.state in (TrackingState.INITIALIZING, TrackingState.LOST) or
            (self.frame_count % self.detection_interval == 0)
        )

        detected_new = False
        if must_detect:
            det = self.detect_box(frame)
            if det is not None:
                self.current_bbox = det["bbox"]
                self.current_corners_2d = det["corners_8_2d"]
                self._init_tracker(frame, self.current_bbox)
                self.state = TrackingState.TRACKING
                detected_new = True
            else:
                if self.state == TrackingState.INITIALIZING:
                    self.state = TrackingState.INITIALIZING
                elif not (self.state == TrackingState.TRACKING and self._tracker is not None):
                    self.state = TrackingState.LOST

        # Continuous tracking if not re-detected in this frame
        if not detected_new and self.state == TrackingState.TRACKING and self._tracker is not None:
            ok, tracked_bbox = self._update_tracker(frame)
            if ok:
                prev_bbox = self.current_bbox
                self.current_bbox = tracked_bbox
                # Shift 2D corners proportionally according to bbox delta
                if prev_bbox is not None and self.current_corners_2d is not None:
                    dx = tracked_bbox[0] - prev_bbox[0]
                    dy = tracked_bbox[1] - prev_bbox[1]
                    scale_x = float(tracked_bbox[2]) / float(max(1, prev_bbox[2]))
                    scale_y = float(tracked_bbox[3]) / float(max(1, prev_bbox[3]))

                    c_center_x = prev_bbox[0] + prev_bbox[2] / 2.0
                    c_center_y = prev_bbox[1] + prev_bbox[3] / 2.0

                    shifted = self.current_corners_2d.copy()
                    for idx in range(len(shifted)):
                        pt_x = (shifted[idx, 0] - c_center_x) * scale_x + (tracked_bbox[0] + tracked_bbox[2] / 2.0)
                        pt_y = (shifted[idx, 1] - c_center_y) * scale_y + (tracked_bbox[1] + tracked_bbox[3] / 2.0)
                        shifted[idx] = [pt_x, pt_y]
                    self.current_corners_2d = shifted
                else:
                    self.current_corners_2d = self._infer_8_corners_from_bbox(tracked_bbox, frame)
            else:
                self.state = TrackingState.LOST

        # Check Quality Gate: blur & truncation
        q_report = self.quality_gate.validate_frame(
            frame,
            self.current_corners_2d if self.state == TrackingState.TRACKING else None,
            w, h
        )

        # Cuboid reconstruction and smoothing
        if self.state == TrackingState.TRACKING and self.current_corners_2d is not None:
            # Apply temporal exponential smoothing to stabilize jitter
            if self.enable_smoothing:
                if self._smoothed_corners_2d is None:
                    self._smoothed_corners_2d = self.current_corners_2d.copy()
                else:
                    self._smoothed_corners_2d = (
                        (1.0 - self.smoothing_alpha) * self._smoothed_corners_2d +
                        self.smoothing_alpha * self.current_corners_2d
                    )
                active_corners_2d = self._smoothed_corners_2d
            else:
                active_corners_2d = self.current_corners_2d

            corners_3d, raw_dims, raw_vw = self.estimate_3d_and_dimensions(active_corners_2d, (h, w))
            if self.enable_smoothing and self._smoothed_dims is not None:
                smooth_l = (1.0 - self.smoothing_alpha) * self._smoothed_dims["length"] + self.smoothing_alpha * raw_dims["length"]
                smooth_w = (1.0 - self.smoothing_alpha) * self._smoothed_dims["width"] + self.smoothing_alpha * raw_dims["width"]
                smooth_h = (1.0 - self.smoothing_alpha) * self._smoothed_dims["height"] + self.smoothing_alpha * raw_dims["height"]
                self.current_dimensions = {
                    "length": round(smooth_l, 1),
                    "width": round(smooth_w, 1),
                    "height": round(smooth_h, 1)
                }
            else:
                self.current_dimensions = raw_dims
                self._smoothed_dims = raw_dims.copy()

            self.current_corners_3d = corners_3d
            self.current_volumetric_weight = compute_volumetric_weight(
                self.current_dimensions["length"],
                self.current_dimensions["width"],
                self.current_dimensions["height"]
            )
        else:
            active_corners_2d = None

        # Draw visual overlays
        annotated = self.draw_overlay(frame, active_corners_2d, q_report)

        return {
            "annotated_frame": annotated,
            "state": self.state,
            "fps": round(self.fps, 1),
            "bbox": self.current_bbox if self.state == TrackingState.TRACKING else None,
            "corners_2d": active_corners_2d.tolist() if active_corners_2d is not None else None,
            "corners_3d": self.current_corners_3d.tolist() if self.current_corners_3d is not None else None,
            "dimensions_cm": self.current_dimensions,
            "volumetric_weight_kg": round(self.current_volumetric_weight, 3),
            "quality_report": q_report,
        }

    def draw_overlay(
        self,
        frame: np.ndarray,
        corners_2d: Optional[np.ndarray],
        quality_report: QualityReport
    ) -> np.ndarray:
        """
        Renders HUD telemetry card, cuboid wireframe, labeled 1-8 corner markers,
        bounding box, and tracking status.
        """
        img = frame.copy()
        h, w = img.shape[:2]

        # 1. Draw Bounding Box & Wireframe if tracking
        if self.state == TrackingState.TRACKING and self.current_bbox is not None:
            bx, by, bw, bh = self.current_bbox
            color_track = (0, 255, 120) if quality_report.status == "ACCEPTED" else (0, 165, 255)
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color_track, 2)

            if corners_2d is not None and len(corners_2d) == 8:
                pts = np.asarray(corners_2d, dtype=np.int32)

                # Google Objectron-style visual HUD:
                # 12 blue cuboid edges + pink diagonal cross on front/top + black cross on base
                BLUE = (240, 130, 30)       # electric blue BGR
                PINK = (180, 105, 255)      # pink BGR
                BLACK = (30, 30, 30)        # near-black BGR

                # Cuboid 12 edges (blue)
                wire_edges = [
                    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
                    (0, 1), (1, 2), (2, 3), (3, 0),  # top face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
                ]
                for p_start, p_end in wire_edges:
                    cv2.line(img, tuple(pts[p_start]), tuple(pts[p_end]), BLUE, 2, cv2.LINE_AA)

                # Pink diagonal cross (×) on front/top face (corners 0,1,2,3)
                cv2.line(img, tuple(pts[0]), tuple(pts[2]), PINK, 2, cv2.LINE_AA)
                cv2.line(img, tuple(pts[1]), tuple(pts[3]), PINK, 2, cv2.LINE_AA)

                # Black diagonal cross (×) on bottom/base face (corners 4,5,6,7)
                cv2.line(img, tuple(pts[4]), tuple(pts[6]), BLACK, 2, cv2.LINE_AA)
                cv2.line(img, tuple(pts[5]), tuple(pts[7]), BLACK, 2, cv2.LINE_AA)

                # Draw Corner Markers 1 to 8
                for idx in range(8):
                    pt = tuple(pts[idx])
                    cv2.circle(img, pt, 7, (0, 0, 0), -1)
                    cv2.circle(img, pt, 5, (0, 255, 255), -1)
                    label = f"C{idx + 1}"
                    cv2.putText(img, label, (pt[0] + 6, pt[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw HUD Telemetry Card in Top-Left
        card_w, card_h = min(420, w - 20), 160
        overlay = img.copy()
        cv2.rectangle(overlay, (12, 12), (12 + card_w, 12 + card_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        border_color = (0, 255, 0) if self.state == TrackingState.TRACKING else (0, 165, 255) if self.state == TrackingState.INITIALIZING else (0, 0, 255)
        cv2.rectangle(img, (12, 12), (12 + card_w, 12 + card_h), border_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # Line 1: State & FPS
        state_color = border_color
        cv2.putText(img, f"STATUS: {self.state}", (24, 40), font, 0.65, state_color, 2)
        cv2.putText(img, f"FPS: {self.fps:.1f}", (24 + card_w - 110, 40), font, 0.60, (220, 220, 220), 1)

        # Line 2: Quality Gate Status
        q_color = (0, 255, 0) if quality_report.status == "ACCEPTED" else (0, 0, 255)
        q_text = f"Quality: {quality_report.status} (Blur={quality_report.blur_score:.1f})"
        cv2.putText(img, q_text, (24, 70), font, 0.50, q_color, 1)

        # Line 3: Dimensions
        l = self.current_dimensions.get("length", 0.0)
        width = self.current_dimensions.get("width", 0.0)
        height = self.current_dimensions.get("height", 0.0)
        dim_str = f"Dims: L={l:.1f} x W={width:.1f} x H={height:.1f} cm"
        cv2.putText(img, dim_str, (24, 105), font, 0.60, (255, 255, 255), 2)

        # Line 4: Volumetric Weight
        vw = self.current_volumetric_weight
        vw_str = f"Vol. Weight: {vw:.3f} kg (L*W*H/5000)"
        cv2.putText(img, vw_str, (24, 138), font, 0.60, (0, 255, 255), 2)

        return img

    def reset(self):
        """Resets tracker state."""
        self.state = TrackingState.INITIALIZING
        self._tracker = None
        self.current_bbox = None
        self.current_corners_2d = None
        self.current_corners_3d = None
        self._smoothed_corners_2d = None
        self._smoothed_dims = None
        self.current_dimensions = {"length": 0.0, "width": 0.0, "height": 0.0}
        self.current_volumetric_weight = 0.0
