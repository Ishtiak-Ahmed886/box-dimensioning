"""Automated tests for OpenCV Camera Calibration (F1, A1)."""

import os
import glob
import pytest
import numpy as np

from box_dimensioning.calibration.camera_calibrator import CameraCalibrator


def test_calibration_rms_reprojection_error():
    """Verify Acceptance Criterion A1: RMS reprojection error <= 0.6 px on OpenCV sample set."""
    calib_dir = "data/calibration"
    images = sorted(glob.glob(os.path.join(calib_dir, "left*.jpg")))
    assert len(images) >= 3, f"At least 3 chessboard images required in {calib_dir}"

    calibrator = CameraCalibrator()
    res = calibrator.calibrate(images)

    rms = res["rms_reprojection_error_px"]
    print(f"\n[Test A1] Measured RMS reprojection error: {rms:.4f} px (threshold <= 0.6 px)")

    assert res["meets_acceptance_criterion_A1"] is True
    assert rms <= 0.60, f"RMS error {rms} exceeds 0.6 px"
    assert calibrator.camera_matrix is not None
    assert calibrator.dist_coeffs is not None


def test_undistortion_preserves_dimensions():
    """Verify lens undistortion runs and maintains image shape."""
    calib_dir = "data/calibration"
    images = sorted(glob.glob(os.path.join(calib_dir, "left*.jpg")))
    calibrator = CameraCalibrator()
    calibrator.calibrate(images)

    import cv2
    img = cv2.imread(images[0])
    undistorted = calibrator.undistort(img)

    assert undistorted.shape == img.shape
    assert not np.array_equal(undistorted, img), "Undistorted image should differ from distorted original"
