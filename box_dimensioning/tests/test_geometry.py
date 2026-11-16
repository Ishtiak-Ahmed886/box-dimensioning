"""Automated unit tests for 3D Box geometry and Sutherland-Hodgman 3D IoU."""

import pytest
import numpy as np

from box_dimensioning.dataset.box_geometry import Box, IoU, compute_volumetric_weight


def test_box_creation_and_properties():
    """Verify 3D Box creation, dimensions sorting, volume, and volumetric weight."""
    # Box: 30cm x 20cm x 10cm (0.3m x 0.2m x 0.1m)
    scale = np.array([0.30, 0.20, 0.10])
    rot = np.eye(3)
    trans = np.array([1.0, 2.0, 3.0])

    box = Box.from_transformation(rot, trans, scale)

    assert abs(box.volume - 0.006) < 1e-6
    l, w, h = box.dimensions_cm
    assert abs(l - 30.0) < 1e-6
    assert abs(w - 20.0) < 1e-6
    assert abs(h - 10.0) < 1e-6

    # Volumetric weight = 30 * 20 * 10 / 5000 = 6000 / 5000 = 1.2 kg
    vw = box.volumetric_weight_kg
    assert abs(vw - 1.2) < 1e-6


def test_sutherland_hodgman_exact_iou():
    """Verify exact Sutherland-Hodgman 3D IoU on identical, partial overlap, and disjoint boxes."""
    scale = np.array([0.20, 0.15, 0.10])
    b1 = Box.from_transformation(np.eye(3), np.zeros(3), scale)

    # 1. Self-IoU must be 1.0
    iou_self = IoU(b1, b1).iou()
    assert abs(iou_self - 1.0) < 1e-3, f"Self-IoU expected 1.0, got {iou_self}"

    # 2. Shifted by half of X-dimension (0.1m)
    # Overlapping volume = (0.1) * (0.15) * (0.10) = 0.0015
    # Union volume = 0.003 + 0.003 - 0.0015 = 0.0045 => IoU = 0.0015 / 0.0045 = 1/3 ~ 0.3333
    b2 = Box.from_transformation(np.eye(3), np.array([0.10, 0.0, 0.0]), scale)
    iou_shift = IoU(b1, b2).iou()
    assert abs(iou_shift - (1.0 / 3.0)) < 0.05, f"Expected ~0.333, got {iou_shift}"

    # 3. Disjoint box
    b3 = Box.from_transformation(np.eye(3), np.array([5.0, 5.0, 5.0]), scale)
    iou_disjoint = IoU(b1, b3).iou()
    assert iou_disjoint == 0.0


def test_box_fit_reconstruction():
    """Verify Box.fit reconstructs original 9-DOF parameters from noisy vertices."""
    scale_gt = np.array([0.25, 0.18, 0.08])
    rot_gt = np.array([
        [0.866, -0.5, 0.0],
        [0.5, 0.866, 0.0],
        [0.0, 0.0, 1.0]
    ])
    trans_gt = np.array([0.2, -0.1, 0.5])

    box_gt = Box.from_transformation(rot_gt, trans_gt, scale_gt)
    vertices_clean = box_gt.vertices

    # Reconstruct
    rot_est, trans_est, scale_est = Box.fit(vertices_clean)
    assert np.allclose(scale_est, scale_gt, atol=1e-3)
    assert np.allclose(trans_est, trans_gt, atol=1e-3)
