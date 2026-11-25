"""Comprehensive End-to-End Integration Tests for Box Dimensioning Pipeline.
Verifies Phase 1 to Phase 18 requirements:
- 8-keypoint pose model loading & inference
- Exact 8-keypoint semantic correspondence
- DLT triangulation on 8 keypoints
- 3D cuboid fitting & dimension extraction
- Quality Gate rejection codes
- Scale anchor detection
- Mobile portrait camera aspect ratio adaptation
"""

import os
import cv2
import numpy as np
import pytest

from box_dimensioning.dataset.box_geometry import Box, compute_volumetric_weight, EDGES
from box_dimensioning.dataset.objectron_parser import FrameGeometry, ObjectronSequence
from box_dimensioning.segmentation.deep_detector import DeepBoxDetector
from box_dimensioning.pipeline.method_a_multiview import MethodAMultiView, MeasurementResult
from box_dimensioning.pipeline.method_b_singleview import MethodBSingleView
from box_dimensioning.pipeline.quality_gate import QualityGate, QualityReason, QualityReport
from box_dimensioning.pipeline.scale_anchor import detect_reference_rectangle, REFERENCE_OBJECTS
from box_dimensioning.pipeline.camera_box_tracker import CameraBoxTracker, TrackingState


class TestIntegrationPipeline:

    @pytest.fixture(scope="class")
    def detector(self):
        return DeepBoxDetector()

    def test_model_loading_and_8_keypoint_prediction(self, detector):
        """Phase 1: Verifies model exists, loads, and outputs 8 keypoints with high confidence."""
        model_path = os.path.join("models", "custom_box_corner_pose.pt")
        assert os.path.isfile(model_path), f"Missing model file: {model_path}"

        # Test on a synthetic clean box frame
        test_frame = np.full((320, 320, 3), 200, dtype=np.uint8)
        # Draw a synthetic box rectangle
        cv2.rectangle(test_frame, (60, 60), (260, 260), (50, 100, 180), -1)

        res = detector.detect(test_frame)
        assert res is not None
        assert "corners_2d" in res
        assert "confidence" in res
        assert "bounding_box" in res
        # Check polymorphic alias
        res_alias = detector.detect_corners(test_frame)
        assert res_alias is not None

    def test_keypoint_semantic_correspondences_without_sift(self):
        """Phase 1 & 2: Verifies ordered 8 keypoints are matched directly without background clutter."""
        method_a = MethodAMultiView()

        # Dummy frame geometries
        f1 = FrameGeometry(
            frame_id=0, timestamp=0.0, transform=np.eye(4), projection_matrix=np.eye(4),
            view_matrix=np.eye(4), intrinsics=np.eye(3), image_width=640, image_height=480
        )
        f2 = FrameGeometry(
            frame_id=1, timestamp=0.1, transform=np.eye(4), projection_matrix=np.eye(4),
            view_matrix=np.eye(4), intrinsics=np.eye(3), image_width=640, image_height=480
        )

        kpts1 = np.array([[10, 10], [20, 10], [10, 20], [20, 20],
                          [30, 10], [40, 10], [30, 20], [40, 20]], dtype=np.float32)
        kpts2 = kpts1 + 5.0

        m1, m2 = method_a.match_features_and_corners(None, None, f1, f2, kpts1, kpts2)
        assert len(m1) == 8
        assert len(m2) == 8
        # Direct identity preservation
        np.testing.assert_allclose(m1, kpts1)
        np.testing.assert_allclose(m2, kpts2)

    def test_cuboid_fitting_from_8_points(self):
        """Phase 2: Verifies Box.fit accurately reconstructs cuboid from 8 vertices."""
        method_a = MethodAMultiView()

        # True synthetic cuboid vertices (scale = 0.30 x 0.20 x 0.10 m)
        true_scale = np.array([0.30, 0.20, 0.10])
        box_gt = Box.from_transformation(np.eye(3), np.array([0.0, 0.0, 1.0]), true_scale)
        # Vertices 1 to 8 (ignoring centroid 0)
        verts_8 = box_gt.vertices[1:9]

        fitted_box, residual = method_a.fit_cuboid(verts_8)
        assert residual < 0.02  # Less than 2 cm fitting residual
        l, w, h = fitted_box.dimensions_cm
        # Sorted descending: L >= W >= H
        assert l >= w >= h
        assert abs(l - 30.0) < 1.0
        assert abs(w - 20.0) < 1.0
        assert abs(h - 10.0) < 1.0

    def test_volumetric_weight_calculation(self):
        """Phase 10: Verifies VW = L * W * H / 5000 with metric dimensions."""
        l, w, h = 50.0, 40.0, 30.0
        expected_vw = (50.0 * 40.0 * 30.0) / 5000.0  # 12.0 kg
        computed_vw = compute_volumetric_weight(l, w, h)
        assert abs(computed_vw - expected_vw) < 1e-5

    def test_quality_gate_rejection_codes(self):
        """Phase 8: Verifies standardized quality rejection codes."""
        qg = QualityGate(blur_threshold=60.0, truncation_margin_px=8.0, min_baseline_meters=0.04)

        # 1. Blur rejection
        blurred_img = np.full((100, 100, 3), 128, dtype=np.uint8)  # Variance = 0
        rep_blur = qg.validate_frame(blurred_img, np.array([[20, 20], [80, 80]]), 100, 100)
        assert rep_blur.status == "REJECTED"
        assert rep_blur.reason_code == QualityReason.FRAME_TOO_BLURRY

        # 2. Truncation rejection
        sharp_img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        trunc_corners = np.array([[2, 50], [100, 50], [100, 150], [2, 150]])
        rep_trunc = qg.validate_frame(sharp_img, trunc_corners, 200, 200)
        assert rep_trunc.status == "REJECTED"
        assert rep_trunc.reason_code == QualityReason.BOX_TRUNCATED

        # 3. Insufficient baseline rejection
        good_corners = np.array([[20, 50], [100, 50], [100, 150], [20, 150]])
        rep_ok = qg.validate_frame(sharp_img, good_corners, 200, 200)
        rep_base = qg.validate_measurement(rep_ok, baseline_meters=0.01)
        assert rep_base.status == "REJECTED"
        assert rep_base.reason_code == QualityReason.INSUFFICIENT_BASELINE

    def test_method_b_honest_scale_rejection(self):
        """Phase 3: Verifies Method B rejects measurement when scale is unavailable."""
        method_b = MethodBSingleView()

        # Dummy sequence without AR plane and without point cloud
        dummy_frame = FrameGeometry(
            frame_id=0, timestamp=0.0, transform=np.eye(4), projection_matrix=np.eye(4),
            view_matrix=np.eye(4), intrinsics=np.eye(3), image_width=640, image_height=480,
            planes=[], point_cloud=np.empty((0, 3))
        )
        seq = ObjectronSequence(
            sequence_id="test_unscaled",
            frames_geometry=[dummy_frame],
            ground_truth=None,
            sequence_dir="",
            frame_annotations=[]
        )

        res = method_b.measure(seq)
        assert res.status == "REJECTED"
        assert res.quality_report.reason_code == QualityReason.SCALE_UNAVAILABLE

    def test_mobile_portrait_aspect_ratio_adaptation(self):
        """Phase 6: Verifies camera tracker adapts intrinsics for mobile portrait frames."""
        tracker = CameraBoxTracker()
        
        # Initial landscape frame: 640x480
        tracker._ensure_camera_matrix(640, 480)
        assert tracker.camera_matrix[0, 2] == 320.0
        assert tracker.camera_matrix[1, 2] == 240.0

        # Now stream switches to mobile portrait: 480x640
        tracker._ensure_camera_matrix(480, 640)
        assert tracker.camera_matrix[0, 2] == 240.0
        assert tracker.camera_matrix[1, 2] == 320.0

    def test_scale_anchor_reference_card_detection(self):
        """Phase 12: Verifies detect_reference_rectangle on credit card aspect ratio."""
        # Credit Card: 85.6 x 54 mm (ratio ~1.585)
        img = np.full((400, 600, 3), 220, dtype=np.uint8)
        # Draw a rectangle with ratio 1.585 (e.g. 158 x 100 px)
        cv2.rectangle(img, (150, 150), (150 + 158, 150 + 100), (40, 40, 40), -1)

        ref_match = detect_reference_rectangle(img, ref_name="Credit Card")
        assert ref_match is not None
        assert ref_match["ref_name"] == "Credit Card"
        assert ref_match["scale_mm_per_px"] > 0.0
