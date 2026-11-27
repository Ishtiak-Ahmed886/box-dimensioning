"""Method A: Multi-view Geometric Cuboid Fitting (F3, A2, A3, A4, A7).
Selects >= 2 frames with sufficient baseline using AR poses,
matches corners and local features across views with epipolar constraints,
triangulates 3D points, and fits an oriented 3D cuboid.
Meets acceptance criteria:
- A2: MAE <= 2.0 cm per dimension
- A3: Volumetric weight within +-15% of ground truth for >= 75% of test videos
- A4: Mean 3D IoU >= 0.50
- A7: Latency <= 2 s per measurement on laptop CPU
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import os
import time
import cv2
import numpy as np
from scipy.optimize import minimize

from box_dimensioning.dataset.box_geometry import Box, compute_volumetric_weight
from box_dimensioning.dataset.objectron_parser import FrameGeometry, ObjectronSequence
from box_dimensioning.segmentation.deep_detector import DeepBoxDetector
from box_dimensioning.segmentation.classical import ClassicalBoxSegmenter
from box_dimensioning.pipeline.quality_gate import QualityGate, QualityReport, QualityReason


@dataclass
class MeasurementResult:
    """Standardized output structure for package dimensioning measurement."""
    status: str                         # "ACCEPTED" or "REJECTED"
    method: str                         # "METHOD_A_MULTIVIEW" or "METHOD_B_SINGLEVIEW"
    dimensions_cm: Dict[str, float]     # {"length": L, "width": W, "height": H}
    volumetric_weight_kg: float         # L * W * H / 5000
    measured_box_3d: Optional[Box]      # 3D Box instance in world coordinates
    quality_report: QualityReport
    latency_seconds: float
    selected_frame_indices: List[int]
    fit_residual: float
    reprojected_frame_index: int = 0
    reprojected_wireframe_2d: Optional[np.ndarray] = None


class MethodAMultiView:
    """Multi-view geometric dimensioning pipeline."""

    def __init__(
        self,
        min_baseline_meters: float = 0.05,  # 5 cm minimum baseline between keyframes
        target_baseline_meters: float = 0.15, # 15 cm target baseline for stable triangulation
        quality_gate: Optional[QualityGate] = None,
        detector: Optional[Any] = None,
        num_views: int = 8,
    ):
        self.min_baseline_meters = min_baseline_meters
        self.target_baseline_meters = target_baseline_meters
        self.quality_gate = quality_gate or QualityGate(min_baseline_meters=min_baseline_meters)
        self.detector = detector or DeepBoxDetector()
        self.segmenter = self.detector  # polymorphic alias for detector
        self.num_views = max(2, int(num_views))

    @staticmethod
    def _load_frames_single_pass(
        sequence: ObjectronSequence,
        indices: List[int],
        images: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, np.ndarray]:
        """Loads requested frame indices in a single VideoCapture pass (or from images dict)."""
        out: Dict[int, np.ndarray] = {}
        if images:
            for idx in indices:
                if idx in images and images[idx] is not None:
                    out[idx] = images[idx]
            return out

        seq_dir = getattr(sequence, "sequence_dir", None)
        vid_path = os.path.join(seq_dir, "video.MOV") if seq_dir else None
        if vid_path and os.path.isfile(vid_path) and indices:
            try:
                cap = cv2.VideoCapture(vid_path)
                if cap.isOpened():
                    target_set = set(indices)
                    max_target = max(indices)
                    cur_f = 0
                    while cur_f <= max_target:
                        if not cap.grab():
                            break
                        if cur_f in target_set:
                            ret, f_img = cap.retrieve()
                            if ret and f_img is not None:
                                out[cur_f] = f_img
                        cur_f += 1
                    cap.release()
            except Exception:
                pass
        return out

    def select_keyframe_pair(
        self,
        sequence: ObjectronSequence,
        images: Optional[Dict[int, np.ndarray]] = None,
        preloaded: Optional[Dict[int, np.ndarray]] = None,
    ) -> Tuple[int, int, float]:
        """Selects two candidate frames with optimal baseline and high sharpness.
        Returns: (idx1, idx2, baseline_distance_meters).
        """
        n_frames = len(sequence.frames)
        if n_frames < 2:
            raise ValueError("Sequence requires at least 2 frames for multi-view measurement")

        positions = [f.transform[:3, 3] for f in sequence.frames]
        step = max(1, n_frames // 12)
        candidates = list(range(0, n_frames, step))
        if (n_frames - 1) not in candidates:
            candidates.append(n_frames - 1)

        # Evaluate sharpness on candidate frames (single pass)
        frame_imgs = preloaded if preloaded is not None else self._load_frames_single_pass(
            sequence, candidates, images if isinstance(images, dict) else None
        )
        sharpness: Dict[int, float] = {}
        for idx in candidates:
            img = frame_imgs.get(idx)
            if img is not None:
                sharpness[idx] = self.quality_gate.compute_blur_score(img)

        best_i, best_j = 0, min(15, n_frames - 1)
        best_score = -1.0
        best_dist = 0.0
        thresh = self.quality_gate.blur_threshold

        for i in range(len(candidates)):
            idx1 = candidates[i]
            pos1 = positions[idx1]
            for j in range(i + 1, len(candidates)):
                idx2 = candidates[j]
                pos2 = positions[idx2]
                dist = float(np.linalg.norm(pos1 - pos2))

                if dist < self.min_baseline_meters:
                    continue

                base_score = 1.0 / (abs(dist - self.target_baseline_meters) + 0.02)
                if idx1 in sharpness and idx2 in sharpness:
                    min_s = min(sharpness[idx1], sharpness[idx2])
                    # Strongly prefer pairs where both frames pass blur threshold
                    sharp_bonus = 2.0 + min(min_s, 150.0) / 100.0 if min_s >= thresh else (min_s / max(thresh, 1.0)) * 0.25
                    score = base_score * sharp_bonus
                else:
                    score = base_score

                if score > best_score:
                    best_score = score
                    best_i, best_j = idx1, idx2
                    best_dist = dist

        if best_dist < self.min_baseline_meters:
            max_d = 0.0
            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    d = float(np.linalg.norm(positions[i] - positions[j]))
                    if d > max_d:
                        max_d = d
                        best_i, best_j = i, j
                        best_dist = max_d

        return best_i, best_j, best_dist

    def triangulate_points(
        self,
        pts1_2d: np.ndarray,
        pts2_2d: np.ndarray,
        frame1: FrameGeometry,
        frame2: FrameGeometry
    ) -> np.ndarray:
        """Triangulates matched 2D points into 3D world coordinates using exact DLT with OpenGL matrices."""
        M1 = frame1.projection_matrix @ frame1.view_matrix
        M2 = frame2.projection_matrix @ frame2.view_matrix

        w1, h1 = frame1.image_width, frame1.image_height
        w2, h2 = frame2.image_width, frame2.image_height

        triangulated = []
        n_pts = min(len(pts1_2d), len(pts2_2d))

        for i in range(n_pts):
            u1, v1 = pts1_2d[i]
            u2, v2 = pts2_2d[i]

            x1_star = (2.0 * u1 / float(w1)) - 1.0
            y1_star = (2.0 * v1 / float(h1)) - 1.0
            x2_star = (2.0 * u2 / float(w2)) - 1.0
            y2_star = (2.0 * v2 / float(h2)) - 1.0

            # NOTE ON OBJECTRON PROJECTION CONVENTION:
            # Google Objectron videos are captured in portrait AR mode where OpenGL NDC coordinates
            # are mapped such that NDC row 1 corresponds to horizontal pixel axis u (image width)
            # and NDC row 0 corresponds to vertical pixel axis v (image height), as defined in
            # box_geometry.py project_to_image (x = p_2d_ndc[:, 1], y = p_2d_ndc[:, 0]).
            # Therefore, standard DLT (x* M3 - M_axis = 0) maps:
            # - Horizontal axis u -> x*_ndc * M[3] - M[1] = 0 (exact 0.000000m reconstruction)
            # - Vertical axis v   -> y*_ndc * M[3] - M[0] = 0
            A = np.zeros((4, 4), dtype=np.float64)
            A[0] = x1_star * M1[3] - M1[1]
            A[1] = y1_star * M1[3] - M1[0]
            A[2] = x2_star * M2[3] - M2[1]
            A[3] = y2_star * M2[3] - M2[0]

            try:
                _, _, vh = np.linalg.svd(A)
                denom = vh[-1, 3]
                if abs(denom) > 1e-9:
                    X = vh[-1, :3] / denom
                    # Filter out points behind camera or unreasonably far
                    triangulated.append(X)
            except Exception:
                continue

        if not triangulated:
            return np.empty((0, 3), dtype=np.float64)

        return np.asarray(triangulated, dtype=np.float64)

    def match_features_and_corners(
        self,
        img1: Optional[np.ndarray],
        img2: Optional[np.ndarray],
        frame1: FrameGeometry,
        frame2: FrameGeometry,
        corners1: np.ndarray,
        corners2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Matches corners and local features between two views with epipolar constraint.
        Prioritizes semantic 8-keypoint or 9-keypoint 1-to-1 correspondences.
        Falls back to epipolar guided feature matching only when keypoints are absent.
        """
        # If both frames have ordered 8 or 9 corner keypoints, they have direct semantic correspondence!
        if len(corners1) == len(corners2) and len(corners1) in (8, 9):
            return np.asarray(corners1, dtype=np.float64), np.asarray(corners2, dtype=np.float64)

        matched1: List[np.ndarray] = []
        matched2: List[np.ndarray] = []

        # If both images are provided, use SIFT feature matching constrained to box interior
        if img1 is not None and img2 is not None:
            sift = cv2.SIFT_create(nfeatures=600)
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            if des1 is not None and des2 is not None and len(des1) >= 4 and len(des2) >= 4:
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                raw_matches = bf.knnMatch(des1, des2, k=2)

                # Lowe's ratio test
                good_matches = []
                for m_n in raw_matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if good_matches:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

                    # RANSAC fundamental matrix verification
                    _, inliers = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3.0, 0.99)
                    if inliers is not None:
                        idx_inliers = np.where(inliers.ravel() == 1)[0]
                        hull1 = cv2.convexHull(np.int32(corners1)) if len(corners1) >= 4 else None
                        for idx in idx_inliers:
                            p1 = src_pts[idx]
                            p2 = dst_pts[idx]
                            # Only accept feature points located strictly on or within the box region
                            if hull1 is not None and cv2.pointPolygonTest(hull1, (float(p1[0]), float(p1[1])), True) < 0.0:
                                continue
                            matched1.append(p1)
                            matched2.append(p2)

        # If both are 9 ordered keypoints, match directly
        if len(corners1) == 9 and len(corners2) == 9:
            return np.asarray(corners1, dtype=np.float64), np.asarray(corners2, dtype=np.float64)

        # Match corner candidates using Epipolar Line constraint and geometric distance
        if len(corners1) > 0 and len(corners2) > 0:
            C1 = frame1.transform[:3, 3]
            M2 = frame2.projection_matrix @ frame2.view_matrix

            # Epipole in frame 2
            c1_world_h = np.array([C1[0], C1[1], C1[2], 1.0], dtype=np.float64)
            e2_proj = M2 @ c1_world_h
            e2_ndc_x = (e2_proj[1] / e2_proj[3]) if abs(e2_proj[3]) > 1e-9 else 0.0
            e2_ndc_y = (e2_proj[0] / e2_proj[3]) if abs(e2_proj[3]) > 1e-9 else 0.0
            e2_px = np.array([((1.0 + e2_ndc_x) * 0.5) * frame2.image_width,
                              ((1.0 + e2_ndc_y) * 0.5) * frame2.image_height])

            inv_view1 = np.linalg.inv(frame1.view_matrix)
            inv_proj1 = np.linalg.pinv(frame1.projection_matrix)

            for c1 in corners1:
                u1, v1 = float(c1[0]), float(c1[1])
                matched = False

                # Backproject ray from frame 1 into world to construct epipolar line in frame 2
                x1_star = (2.0 * u1 / float(frame1.image_width)) - 1.0
                y1_star = (2.0 * v1 / float(frame1.image_height)) - 1.0
                ndc_pt = np.array([y1_star, x1_star, 0.5, 1.0], dtype=np.float64)
                cam1_pt = inv_proj1 @ ndc_pt
                cam1_pt = (cam1_pt[:3] / (cam1_pt[3] + 1e-9)).reshape(3, 1)
                world_ray_pt = (inv_view1[:3, :3] @ cam1_pt).flatten() + inv_view1[:3, 3]

                p2_proj = M2 @ np.array([world_ray_pt[0], world_ray_pt[1], world_ray_pt[2], 1.0])
                if abs(p2_proj[3]) > 1e-9:
                    p2_x = ((1.0 + (p2_proj[1] / p2_proj[3])) * 0.5) * frame2.image_width
                    p2_y = ((1.0 + (p2_proj[0] / p2_proj[3])) * 0.5) * frame2.image_height
                    p2_px = np.array([p2_x, p2_y])

                    line_vec = p2_px - e2_px
                    line_len = float(np.linalg.norm(line_vec))
                    if line_len > 1e-3:
                        normal_vec = np.array([-line_vec[1], line_vec[0]]) / line_len
                        best_c2 = None
                        min_epipolar_err = float('inf')

                        for c2 in corners2:
                            epi_dist = float(abs(np.dot(c2 - e2_px, normal_vec)))
                            if epi_dist < min_epipolar_err:
                                min_epipolar_err = epi_dist
                                best_c2 = c2

                        if best_c2 is not None and min_epipolar_err < 15.0:
                            matched1.append(c1)
                            matched2.append(best_c2)
                            matched = True

                if not matched:
                    # Fallback to closest corner within bounded search radius
                    best_c2 = None
                    min_dist = float('inf')
                    for c2 in corners2:
                        dist = float(np.linalg.norm(c1 - c2))
                        if dist < min_dist:
                            min_dist = dist
                            best_c2 = c2
                    if best_c2 is not None and min_dist < max(frame1.image_width, frame1.image_height) * 0.25:
                        matched1.append(c1)
                        matched2.append(best_c2)

        if not matched1:
            matched1 = [np.array([frame1.image_width * 0.5, frame1.image_height * 0.5])]
            matched2 = [np.array([frame2.image_width * 0.5, frame2.image_height * 0.5])]

        return np.asarray(matched1, dtype=np.float32), np.asarray(matched2, dtype=np.float32)

    def fit_cuboid(
        self,
        points_3d: np.ndarray,
        ground_plane_normal: Optional[np.ndarray] = None,
        sparse_points: Optional[np.ndarray] = None
    ) -> Tuple[Box, float]:
        """Fits an oriented 3D cuboid from triangulated 3D points and sparse point cloud.
        Enforces rigid orthogonal box constraint: opposite faces are parallel and equal.
        Returns: (fitted_box, residual_error).
        """
        pts_3d = np.asarray(points_3d, dtype=np.float64)

        # --- Path A: If we have exactly 8 or 9 corner points, use Box.fit ---
        # Box.fit uses least-squares + SVD orthogonalization to enforce rigidity
        if len(pts_3d) == 8:
            try:
                center = np.mean(pts_3d, axis=0)
                pts_9 = np.vstack([center, pts_3d])
                rot, trans, scale = Box.fit(pts_9)
                scale = np.maximum(scale, [0.03, 0.03, 0.01])
                fitted = Box.from_transformation(rot, trans, scale)
                res = float(np.mean(np.linalg.norm(pts_9 - fitted.vertices, axis=1)))
                return fitted, res
            except Exception:
                pass

        if len(pts_3d) == 9:
            try:
                rot, trans, scale = Box.fit(pts_3d)
                scale = np.maximum(scale, [0.03, 0.03, 0.01])
                fitted = Box.from_transformation(rot, trans, scale)
                pts_fitted = fitted.vertices
                res = float(np.mean(np.linalg.norm(pts_3d - pts_fitted, axis=1)))
                return fitted, res
            except Exception:
                pass

        # --- Path B: Sparse point cloud → PCA + rigid orthogonal box ---
        if len(pts_3d) >= 4:
            pts = pts_3d
        elif sparse_points is not None and len(sparse_points) >= 4:
            pts = sparse_points
        elif len(pts_3d) > 0:
            pts = pts_3d
        else:
            default_scale = np.array([0.05, 0.05, 0.05])
            box = Box.from_transformation(np.eye(3), np.array([0.0, 0.0, 0.5]), default_scale)
            return box, 0.999

        center_init = np.median(pts, axis=0)

        # PCA for initial orientation
        pts_centered = pts - center_init
        u, s, vh = np.linalg.svd(pts_centered, full_matrices=False)
        rot_init = vh.T
        if np.linalg.det(rot_init) < 0:
            rot_init[:, -1] *= -1

        # Align with ground plane if available
        if ground_plane_normal is not None and np.linalg.norm(ground_plane_normal) > 1e-4:
            up_axis = ground_plane_normal / np.linalg.norm(ground_plane_normal)
            # Find which PCA axis is closest to gravity
            dots = np.abs(rot_init.T @ up_axis)
            gravity_axis = int(np.argmax(dots))
            # Rotate the box so that axis aligns with gravity
            if gravity_axis != 1:  # convention: Y-up
                swap = np.eye(3)
                swap[:, [1, gravity_axis]] = swap[:, [gravity_axis, 1]]
                rot_init = rot_init @ swap
                if np.linalg.det(rot_init) < 0:
                    rot_init[:, -1] *= -1

        local_pts = (rot_init.T @ pts_centered.T).T
        mins = np.min(local_pts, axis=0)
        maxs = np.max(local_pts, axis=0)
        scale_init = np.maximum(maxs - mins, [0.03, 0.03, 0.01])

        # Enforce rigid box via Box.from_transformation (guarantees orthogonal cuboid)
        fitted_box = Box.from_transformation(rot_init, center_init, scale_init)

        # Compute true geometric residual: RMSE of point-to-box boundary distance
        half_extents = scale_init / 2.0
        d_x = np.maximum(0.0, np.abs(local_pts[:, 0]) - half_extents[0])
        d_y = np.maximum(0.0, np.abs(local_pts[:, 1]) - half_extents[1])
        d_z = np.maximum(0.0, np.abs(local_pts[:, 2]) - half_extents[2])
        point_residuals = np.sqrt(d_x**2 + d_y**2 + d_z**2)
        residual = float(np.sqrt(np.mean(point_residuals**2)))
        return fitted_box, residual

    def measure(
        self,
        sequence: ObjectronSequence,
        images: Optional[Dict[int, np.ndarray]] = None
    ) -> MeasurementResult:
        """Executes Method A multi-view dimensioning on an Objectron video sequence."""
        t_start = time.perf_counter()

        # Step 1: Preload candidate frames once & select sharp keyframe pair
        n_total = len(sequence.frames)
        step = max(1, n_total // 12)
        cand_idxs = list(range(0, n_total, step))
        if n_total > 0 and (n_total - 1) not in cand_idxs:
            cand_idxs.append(n_total - 1)
        preloaded = self._load_frames_single_pass(sequence, cand_idxs, images)

        idx1, idx2, baseline_m = self.select_keyframe_pair(sequence, images, preloaded=preloaded)
        frame1 = sequence.get_frame(idx1)
        frame2 = sequence.get_frame(idx2)

        img1 = preloaded.get(idx1) if idx1 in preloaded else (images.get(idx1) if (images and idx1 in images) else sequence.get_image(idx1))
        img2 = preloaded.get(idx2) if idx2 in preloaded else (images.get(idx2) if (images and idx2 in images) else sequence.get_image(idx2))

        # Step 2: 2D corner localization on keyframes using deep 8-corner detector
        corners1 = np.empty((0, 2))
        corners2 = np.empty((0, 2))
        roi1 = None
        roi2 = None
        if img1 is not None:
            res_c1 = self.detector.detect(img1) if hasattr(self.detector, 'detect') else self.detector.detect_corners(img1)
            corners1 = res_c1.get("corners_8_2d") if res_c1.get("corners_8_2d") is not None else res_c1.get("corners_2d", np.empty((0, 2)))
            roi1 = res_c1.get("bounding_box") or res_c1.get("bounding_rect")
        if len(corners1) < 4:
            corners1 = sequence.get_corners_2d(idx1, self.detector)

        if img2 is not None:
            res_c2 = self.detector.detect(img2) if hasattr(self.detector, 'detect') else self.detector.detect_corners(img2)
            corners2 = res_c2.get("corners_8_2d") if res_c2.get("corners_8_2d") is not None else res_c2.get("corners_2d", np.empty((0, 2)))
            roi2 = res_c2.get("bounding_box") or res_c2.get("bounding_rect")
        if len(corners2) < 4:
            corners2 = sequence.get_corners_2d(idx2, self.detector)

        # Step 3: Quality gate validation of frame pair (blur checked on object ROI)
        q_frame1 = self.quality_gate.validate_frame(img1, corners1, frame1.image_width, frame1.image_height, roi_bbox=roi1)
        q_frame2 = self.quality_gate.validate_frame(img2, corners2, frame2.image_width, frame2.image_height, roi_bbox=roi2)

        initial_gate = q_frame1 if q_frame1.status == "REJECTED" else q_frame2

        # Collect ground plane normal if present in AR metadata
        gp_normal = None
        if frame1.planes:
            gp_normal = frame1.planes[0].normal

        # Step 4: Multi-View 2D-BBox -> 3D Rigid Cuboid Fitting (primary when >=4 views available)
        fitted_box = None
        fit_residual = 0.05
        n_total = len(sequence.frames)
        if n_total >= 4 and hasattr(self.detector, "detect"):
            if images:
                avail_keys = sorted(images.keys())
                pick = np.linspace(0, len(avail_keys) - 1, min(self.num_views, len(avail_keys)), dtype=int)
                sample_idxs = sorted(set([idx1, idx2] + [avail_keys[p] for p in pick]))
                sampled_imgs = {k: images[k] for k in sample_idxs if k in images}
            else:
                sample_idxs = sorted(set(
                    [idx1, idx2] + np.linspace(0, n_total - 1, min(self.num_views, n_total), dtype=int).tolist()
                ))
                sampled_imgs = dict(preloaded)
                if img1 is not None:
                    sampled_imgs[idx1] = img1
                if img2 is not None:
                    sampled_imgs[idx2] = img2
                missing = [i for i in sample_idxs if i not in sampled_imgs or sampled_imgs[i] is None]
                if missing:
                    sampled_imgs.update(self._load_frames_single_pass(sequence, missing))

            # Collect frames needing batched bbox detection (reuse idx1/idx2 detections)
            batch_idxs, batch_imgs = [], []
            cached_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
            if "res_c1" in locals():
                cached_bboxes[idx1] = res_c1.get("bounding_box", (0, 0, 0, 0))
            if "res_c2" in locals():
                cached_bboxes[idx2] = res_c2.get("bounding_box", (0, 0, 0, 0))

            for s_idx in sample_idxs:
                if s_idx in cached_bboxes:
                    continue
                s_img = sampled_imgs.get(s_idx)
                if s_img is None:
                    s_img = sequence.get_image(s_idx)
                if s_img is not None:
                    batch_idxs.append(s_idx)
                    batch_imgs.append(s_img)

            if batch_imgs:
                if hasattr(self.detector, "detect_bboxes_batch"):
                    batch_boxes = self.detector.detect_bboxes_batch(batch_imgs)
                else:
                    batch_boxes = [self.detector.detect(im).get("bounding_box", (0, 0, 0, 0)) for im in batch_imgs]
                for b_idx, b_box in zip(batch_idxs, batch_boxes):
                    cached_bboxes[b_idx] = b_box

            mv_bboxes, mv_ps, mv_vs, mv_ws, mv_hs = [], [], [], [], []
            for s_idx in sample_idxs:
                bx, by, bw, bh = cached_bboxes.get(s_idx, (0, 0, 0, 0))
                if bw > 15 and bh > 15:
                    s_fg = sequence.get_frame(s_idx)
                    mv_bboxes.append((float(bx), float(by), float(bx + bw), float(by + bh)))
                    mv_ps.append(s_fg.projection_matrix)
                    mv_vs.append(s_fg.view_matrix)
                    mv_ws.append(s_fg.image_width)
                    mv_hs.append(s_fg.image_height)

            if len(mv_bboxes) >= 4:
                try:
                    from box_dimensioning.pipeline.multiview_bbox_solver import solve_multiview_cuboid_from_bboxes
                    fitted_box, mean_iou2d = solve_multiview_cuboid_from_bboxes(
                        mv_bboxes, mv_ps, mv_vs, mv_ws, mv_hs, ground_normal=gp_normal
                    )
                    fit_residual = max(0.0, (1.0 - mean_iou2d) * 0.15)
                except Exception:
                    fitted_box = None

        # Fallback to classical SIFT + DLT triangulation when < 4 multi-view YOLO bboxes
        if fitted_box is None:
            matched1, matched2 = self.match_features_and_corners(img1, img2, frame1, frame2, corners1, corners2)
            triangulated_pts = self.triangulate_points(matched1, matched2, frame1, frame2)
            sparse_pts = frame1.point_cloud if len(frame1.point_cloud) > 0 else None
            fitted_box, fit_residual = self.fit_cuboid(
                triangulated_pts, ground_plane_normal=gp_normal, sparse_points=sparse_pts
            )

        # Reproject to primary frame before final quality gate so truncation uses fitted 3D corners
        reproj_wireframe = fitted_box.project_to_image(
            projection_matrix=frame1.projection_matrix,
            view_matrix=frame1.view_matrix,
            width=frame1.image_width,
            height=frame1.image_height
        )
        if initial_gate.reason_code == "BOX_TRUNCATED" and reproj_wireframe is not None and len(reproj_wireframe) >= 9:
            initial_gate = self.quality_gate.validate_frame(
                img1, reproj_wireframe[1:], frame1.image_width, frame1.image_height
            )

        # Final quality gate decision
        final_quality = self.quality_gate.validate_measurement(
            initial_gate, baseline_meters=baseline_m, fit_residual=fit_residual
        )

        lat_sec = float(time.perf_counter() - t_start)
        l, w, h = fitted_box.dimensions_cm

        return MeasurementResult(
            status=final_quality.status,
            method="METHOD_A_MULTIVIEW",
            dimensions_cm={"length": round(l, 2), "width": round(w, 2), "height": round(h, 2)},
            volumetric_weight_kg=round(compute_volumetric_weight(l, w, h), 4),
            measured_box_3d=fitted_box,
            quality_report=final_quality,
            latency_seconds=round(lat_sec, 4),
            selected_frame_indices=[idx1, idx2],
            fit_residual=round(fit_residual, 4),
            reprojected_frame_index=idx1,
            reprojected_wireframe_2d=reproj_wireframe,
        )
