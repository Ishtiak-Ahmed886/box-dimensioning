"""Method B: Single-view Metrology and Plane-Guided Cuboid Fitting (F4, A5).
Uses a single frame + ground plane (from AR plane metadata or RANSAC estimated)
to perform single-view metrology, scaled with sparse AR feature points to fit an oriented 3D box.
Evaluated on the same metrics as Method A (accuracy, failure modes, runtime) for Criterion A5.
"""

from typing import Tuple, Dict, Optional, List, Any
import time
import cv2
import numpy as np

from box_dimensioning.dataset.box_geometry import Box, compute_volumetric_weight
from box_dimensioning.dataset.objectron_parser import FrameGeometry, ObjectronSequence
from box_dimensioning.segmentation.deep_detector import DeepBoxDetector
from box_dimensioning.segmentation.classical import ClassicalBoxSegmenter
from box_dimensioning.pipeline.quality_gate import QualityGate, QualityReport, QualityReason
from box_dimensioning.pipeline.method_a_multiview import MeasurementResult
from box_dimensioning.pipeline.scale_anchor import detect_reference_rectangle


class MethodBSingleView:
    """Single-view metrology dimensioning pipeline."""

    def __init__(self, quality_gate: Optional[QualityGate] = None, detector: Optional[Any] = None):
        self.quality_gate = quality_gate or QualityGate()
        self.detector = detector or DeepBoxDetector()
        self.segmenter = self.detector  # polymorphic alias

    def select_best_frame(
        self,
        sequence: ObjectronSequence,
        images: Optional[Dict[int, np.ndarray]] = None
    ) -> int:
        """Selects the single clearest, most centered frame in the sequence."""
        n_frames = len(sequence.frames)
        if n_frames == 0:
            return 0

        # Pick frame near center of trajectory with good tracking
        center_idx = n_frames // 2
        best_idx = center_idx
        best_score = -1.0

        stride = max(1, n_frames // 20)
        for i in range(0, n_frames, stride):
            frame = sequence.get_frame(i)
            img = images.get(i) if images else None
            blur_score = self.quality_gate.compute_blur_score(img) if img is not None else 100.0

            # Prefer frames with detected planes and good feature points
            plane_bonus = 50.0 if len(frame.planes) > 0 else 0.0
            pt_bonus = min(30.0, float(len(frame.point_cloud)))

            score = blur_score + plane_bonus + pt_bonus
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    def estimate_ground_plane(self, frame: FrameGeometry) -> Tuple[np.ndarray, float]:
        """Retrieves AR detected plane or estimates ground plane from sparse point cloud via RANSAC.
        Returns: (normal_3d, d) such that n.T * X + d = 0.
        """
        # If ARKit/ARCore plane is available, use it
        if frame.planes:
            p = frame.planes[0]
            normal = p.normal / (np.linalg.norm(p.normal) + 1e-9)
            center = p.center
            d = -float(np.dot(normal, center))
            return normal, d

        # Fallback: RANSAC plane estimation from sparse 3D point cloud
        pts = frame.point_cloud
        if len(pts) >= 6:
            best_inliers = 0
            best_plane = (np.array([0.0, 1.0, 0.0]), 0.0)
            for _ in range(50):
                sample_idx = np.random.choice(len(pts), 3, replace=False)
                p1, p2, p3 = pts[sample_idx]
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm = np.linalg.norm(n)
                if norm < 1e-6:
                    continue
                n = n / norm
                d = -np.dot(n, p1)

                dists = np.abs(pts @ n + d)
                inliers = np.sum(dists < 0.03)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_plane = (n, float(d))
            if best_inliers >= 4:
                return best_plane

        # Default horizontal plane at y = -0.15m in camera coordinate system
        cam_pos = frame.transform[:3, 3]
        up = frame.transform[:3, 1]  # camera Y-up
        d = -float(np.dot(up, cam_pos - up * 0.4))
        return up, d

    def raycast_to_plane(
        self,
        pts_2d: np.ndarray,
        frame: FrameGeometry,
        plane_normal: np.ndarray,
        plane_d: float
    ) -> np.ndarray:
        """Projects 2D image coordinates onto 3D ground plane via ray casting."""
        if len(pts_2d) == 0:
            return np.empty((0, 3))

        inv_k = np.linalg.inv(frame.intrinsics.reshape(3, 3))
        cam_center = frame.transform[:3, 3]
        r_c2w = frame.transform[:3, :3]

        homo_2d = np.concatenate([pts_2d, np.ones((len(pts_2d), 1))], axis=1).T  # 3xN
        rays_cam = inv_k @ homo_2d
        rays_cam = rays_cam / (np.linalg.norm(rays_cam, axis=0, keepdims=True) + 1e-9)
        rays_world = (r_c2w @ rays_cam).T  # Nx3

        # Ray-plane intersection: C + t * ray
        # n.T * (C + t * r) + d = 0 => t = (-d - n.T * C) / (n.T * r)
        denom = rays_world @ plane_normal
        numer = -plane_d - np.dot(plane_normal, cam_center)

        valid_mask = np.abs(denom) > 1e-6
        ts = np.zeros(len(rays_world))
        ts[valid_mask] = numer / denom[valid_mask]

        # Only accept rays intersecting in front of camera (t > 0)
        positive_mask = (ts > 0.05) & (ts < 4.0)
        if not np.any(positive_mask):
            return np.empty((0, 3))

        pts_3d = cam_center + (rays_world[positive_mask] * ts[positive_mask, np.newaxis])
        return pts_3d

    def measure(
        self,
        sequence: ObjectronSequence,
        images: Optional[Dict[int, np.ndarray]] = None
    ) -> MeasurementResult:
        """Executes Method B single-view dimensioning on an Objectron video sequence."""
        t_start = time.perf_counter()

        frame_idx = self.select_best_frame(sequence, images)
        frame = sequence.get_frame(frame_idx)
        img = images.get(frame_idx) if (images and frame_idx in images) else sequence.get_image(frame_idx)

        # Detect 2D corners using deep detector
        corners = np.empty((0, 2))
        if img is not None:
            res_c = self.detector.detect(img) if hasattr(self.detector, 'detect') else self.detector.detect_corners(img)
            corners = res_c.get("corners_8_2d") if res_c.get("corners_8_2d") is not None else res_c.get("corners_2d", np.empty((0, 2)))
        if len(corners) < 4:
            corners = sequence.get_corners_2d(frame_idx, self.detector)

        # Quality gate validation
        q_report = self.quality_gate.validate_frame(img, corners, frame.image_width, frame.image_height)

        # Check for scale anchor (reference card or A4 paper)
        ref_match = None
        if img is not None:
            ref_match = detect_reference_rectangle(img)

        # Honest scale check: If no reference card, no AR plane, and no sparse point cloud exist:
        if ref_match is None and len(frame.planes) == 0 and len(frame.point_cloud) < 4:
            q_scale = QualityReport(
                status="REJECTED",
                reason_code=QualityReason.SCALE_UNAVAILABLE,
                blur_score=q_report.blur_score,
                is_blurry=q_report.is_blurry,
                is_truncated=q_report.is_truncated,
                details={"message": "Scale unavailable for single-view metrology. Place an ISO ID-1 credit card or A4 reference sheet in view."}
            )
            lat_sec = float(time.perf_counter() - t_start)
            return MeasurementResult(
                status="REJECTED",
                method="METHOD_B_SINGLEVIEW",
                dimensions_cm={"length": 0.0, "width": 0.0, "height": 0.0},
                volumetric_weight_kg=0.0,
                measured_box_3d=None,
                quality_report=q_scale,
                latency_seconds=round(lat_sec, 4),
                selected_frame_indices=[frame_idx],
                fit_residual=0.999,
                reprojected_frame_index=frame_idx,
                reprojected_wireframe_2d=None
            )

        # Ground plane metrology
        normal, d = self.estimate_ground_plane(frame)
        base_pts_3d = self.raycast_to_plane(corners, frame, normal, d)

        # Incorporate sparse AR feature points
        sparse_pts = frame.point_cloud if len(frame.point_cloud) > 0 else np.empty((0, 3))

        # Estimate base footprint and height
        if len(base_pts_3d) >= 3:
            pts_cluster = base_pts_3d
            center_base = np.median(base_pts_3d, axis=0)

            # Footprint dimensions along horizontal axes
            u, s, vh = np.linalg.svd(pts_cluster - center_base, full_matrices=False)
            rot = vh.T
            if np.linalg.det(rot) < 0:
                rot[:, -1] *= -1

            local_pts = (rot.T @ (pts_cluster - center_base).T).T
            dim_x = float(np.percentile(local_pts[:, 0], 95) - np.percentile(local_pts[:, 0], 5))
            dim_z = float(np.percentile(local_pts[:, 1], 95) - np.percentile(local_pts[:, 1], 5))

            # Height estimation via sparse points above ground plane or vertical ray unprojection
            dim_y = 0.0
            if len(sparse_pts) > 0:
                heights = np.abs((sparse_pts - center_base) @ normal)
                valid_h = heights[(heights > 0.01) & (heights < 0.45)]
                if len(valid_h) >= 2:
                    dim_y = float(np.percentile(valid_h, 90))

            if dim_y <= 0.01:
                # Geometric unprojection from 2D image corner span and camera distance
                cam_pos = frame.transform[:3, 3]
                dist_to_cam = float(np.linalg.norm(center_base - cam_pos))
                f_y = float(frame.intrinsics[1, 1]) if frame.intrinsics.ndim == 2 else 1000.0
                h_px = float(np.ptp(corners[:, 1])) if len(corners) > 0 else 0.0
                dim_y = float((h_px * dist_to_cam) / max(100.0, f_y)) * 0.7

            scale = np.maximum([dim_x, dim_y, dim_z], [0.03, 0.02, 0.03])
            center_3d = center_base + normal * (scale[1] / 2.0)
            fitted_box = Box.from_transformation(rot, center_3d, scale)

            # Calculate true geometric residual: RMSE of base points to fitted rectangle boundary
            half_x, half_z = scale[0] / 2.0, scale[2] / 2.0
            res_x = np.maximum(0.0, np.abs(local_pts[:, 0]) - half_x)
            res_z = np.maximum(0.0, np.abs(local_pts[:, 1]) - half_z)
            fit_residual = float(np.sqrt(np.mean(res_x**2 + res_z**2)))
        else:
            # Fallback based on sparse point cloud bounds if available
            if len(sparse_pts) >= 6:
                center_3d = np.median(sparse_pts, axis=0)
                diffs = np.percentile(sparse_pts, 90, axis=0) - np.percentile(sparse_pts, 10, axis=0)
                scale = np.maximum(diffs, [0.05, 0.03, 0.02])
                rot = frame.transform[:3, :3]
                fitted_box = Box.from_transformation(rot, center_3d, scale)
                # Compute point dispersion residual
                dists = np.linalg.norm(sparse_pts - center_3d, axis=1)
                fit_residual = float(np.std(dists))
            else:
                # Honest failure: neither ground plane intersection nor point cloud provided enough geometry
                center_3d = frame.transform[:3, 3] + frame.transform[:3, 2] * 0.5
                scale = np.array([0.01, 0.01, 0.01])
                rot = frame.transform[:3, :3]
                fitted_box = Box.from_transformation(rot, center_3d, scale)
                fit_residual = 0.999  # Triggers Quality Gate rejection for insufficient geometry

        final_quality = self.quality_gate.validate_measurement(q_report, fit_residual=fit_residual)
        lat_sec = float(time.perf_counter() - t_start)
        l, w, h = fitted_box.dimensions_cm

        reproj_wireframe = fitted_box.project_to_image(
            projection_matrix=frame.projection_matrix,
            view_matrix=frame.view_matrix,
            width=frame.image_width,
            height=frame.image_height
        )

        return MeasurementResult(
            status=final_quality.status,
            method="METHOD_B_SINGLEVIEW",
            dimensions_cm={"length": round(l, 2), "width": round(w, 2), "height": round(h, 2)},
            volumetric_weight_kg=round(compute_volumetric_weight(l, w, h), 4),
            measured_box_3d=fitted_box,
            quality_report=final_quality,
            latency_seconds=round(lat_sec, 4),
            selected_frame_indices=[frame_idx],
            fit_residual=round(fit_residual, 4),
            reprojected_frame_index=frame_idx,
            reprojected_wireframe_2d=reproj_wireframe,
        )
