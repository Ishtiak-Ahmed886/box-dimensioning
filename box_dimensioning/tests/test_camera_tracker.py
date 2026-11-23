"""Unit tests for real-time camera box detection and continuous tracking pipeline."""

import pytest
import numpy as np
import cv2

from box_dimensioning.pipeline.camera_box_tracker import (
    CameraBoxTracker,
    TrackingState,
    OpticalFlowTracker,
)
from box_dimensioning.app.live_cam_tracker import generate_synthetic_box_frame


def test_synthetic_frame_generation():
    """Verifies synthetic frame generation produces valid BGR images."""
    frame = generate_synthetic_box_frame(width=640, height=480, box_pos=(200, 150), box_size=(200, 140))
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_box_clutter_filter():
    """Tests geometric filter distinguishes box-like shapes from cups/distractors."""
    tracker = CameraBoxTracker()
    img_shape = (480, 640)

    # 1. Box-like rectangular contour
    rect_box = np.array([
        [[200, 150]],
        [[400, 150]],
        [[400, 300]],
        [[200, 300]]
    ], dtype=np.int32)
    is_box, approx, meta = tracker.is_box_like_contour(rect_box, img_shape)
    assert is_box is True
    assert meta["solidity"] >= 0.78
    assert 4 <= meta["vertices"] <= 8

    # 2. Highly circular contour (representing a coffee cup)
    circle_pts = []
    for deg in range(0, 360, 10):
        rad = np.deg2rad(deg)
        circle_pts.append([[int(300 + 80 * np.cos(rad)), int(240 + 80 * np.sin(rad))]])
    cup_contour = np.array(circle_pts, dtype=np.int32)
    is_box_cup, _, cup_meta = tracker.is_box_like_contour(cup_contour, img_shape)
    assert is_box_cup is False
    assert cup_meta["reason"] in ("too_circular_cup_bottle", "vertex_count_out_of_range")

    # 3. Very tiny contour (area too small)
    tiny_contour = np.array([
        [[10, 10]], [[15, 10]], [[15, 15]], [[10, 15]]
    ], dtype=np.int32)
    is_box_tiny, _, tiny_meta = tracker.is_box_like_contour(tiny_contour, img_shape)
    assert is_box_tiny is False
    assert tiny_meta["reason"] == "area_too_small"


def test_camera_box_tracker_lifecycle_and_state_machine():
    """Verifies INITIALIZING -> TRACKING state transition on frame with a box."""
    tracker = CameraBoxTracker(
        default_distance_cm=60.0,
        detection_interval=10,
        tracker_type="mil",
        enable_smoothing=True
    )
    assert tracker.state == TrackingState.INITIALIZING

    # Process first synthetic frame containing box
    frame1 = generate_synthetic_box_frame(box_pos=(200, 150))
    res1 = tracker.process_frame(frame1)

    assert res1["state"] == TrackingState.TRACKING
    assert res1["bbox"] is not None
    assert res1["corners_2d"] is not None
    assert len(res1["corners_2d"]) == 8
    assert res1["corners_3d"] is not None
    assert len(res1["corners_3d"]) == 8

    # Verify metric dimensions
    dims = res1["dimensions_cm"]
    assert dims["length"] > 0
    assert dims["width"] > 0
    assert dims["height"] > 0
    assert res1["volumetric_weight_kg"] > 0

    # Process second frame with shifted box position (tests tracking continuity)
    frame2 = generate_synthetic_box_frame(box_pos=(206, 150))
    res2 = tracker.process_frame(frame2)
    assert res2["state"] == TrackingState.TRACKING
    assert res2["corners_2d"] is not None


def test_optical_flow_tracker_fallback():
    """Verifies OpticalFlowTracker initializes and tracks shifted bounding box."""
    oft = OpticalFlowTracker(max_features=25)
    frame1 = generate_synthetic_box_frame(box_pos=(180, 140))
    bbox1 = (180, 140, 200, 150)

    ok_init = oft.init(frame1, bbox1)
    assert ok_init is True

    frame2 = generate_synthetic_box_frame(box_pos=(190, 140))
    ok_track, tracked_bbox = oft.update(frame2)
    assert ok_track is True
    assert tracked_bbox is not None
    # x should have shifted rightwards
    assert tracked_bbox[0] >= bbox1[0] - 5


def test_cuboid_projection_and_volumetric_weight():
    """Verifies 3D cuboid projection and volumetric weight formula L*W*H/5000."""
    tracker = CameraBoxTracker(default_distance_cm=50.0)
    frame = generate_synthetic_box_frame(box_pos=(220, 160), box_size=(180, 130))
    res = tracker.process_frame(frame)

    assert res["state"] == TrackingState.TRACKING
    d = res["dimensions_cm"]
    expected_vw = (d["length"] * d["width"] * d["height"]) / 5000.0
    assert pytest.approx(res["volumetric_weight_kg"], rel=1e-2) == round(expected_vw, 3)

    # Verify visual overlay rendering produced valid output
    annotated = res["annotated_frame"]
    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8


def test_tracker_reset():
    """Verifies tracker reset restores initial state."""
    tracker = CameraBoxTracker()
    frame = generate_synthetic_box_frame()
    tracker.process_frame(frame)
    assert tracker.state == TrackingState.TRACKING

    tracker.reset()
    assert tracker.state == TrackingState.INITIALIZING
    assert tracker.current_bbox is None
    assert tracker.current_corners_2d is None
