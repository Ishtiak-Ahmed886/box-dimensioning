"""Unit tests for Multi-View 2D-BBox to 3D Rigid Cuboid Solver."""

import numpy as np
from scipy.spatial.transform import Rotation

from box_dimensioning.dataset.box_geometry import Box
from box_dimensioning.pipeline.multiview_bbox_solver import (
    solve_multiview_cuboid_from_bboxes,
    _project_corners_batch,
)


def _make_opengl_proj(fx: float, fy: float, w: int, h: int, near: float = 0.1, far: float = 10.0) -> np.ndarray:
    """Builds a minimal OpenGL-style projection matrix matching Objectron NDC."""
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 2.0 * fy / h
    P[1, 1] = 2.0 * fx / w
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2.0 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


def _make_look_at_view(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Builds a 4x4 view matrix looking from cam_pos toward target."""
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    R = np.vstack([right, true_up, -forward])
    V = np.eye(4, dtype=np.float64)
    V[:3, :3] = R
    V[:3, 3] = -R @ cam_pos
    return V


def test_multiview_bbox_solver_recovers_synthetic_cuboid():
    """Recovers exact dimensions (within 1.5 cm) from 8 multi-view 2D bboxes."""
    true_center = np.array([0.0, 0.0, -0.8])
    true_scale = np.array([0.24, 0.18, 0.08])  # 24 x 18 x 8 cm
    true_rot = Rotation.from_euler("y", 25.0, degrees=True).as_matrix()
    gt_box = Box.from_transformation(true_rot, true_center, true_scale)

    w, h = 720, 960
    P = _make_opengl_proj(800.0, 800.0, w, h)

    bboxes, ps, vs, ws, hs = [], [], [], [], []
    angles = np.linspace(-55.0, 55.0, 8)
    for ang in angles:
        rad = np.radians(ang)
        cam_pos = np.array([0.75 * np.sin(rad), 0.25, -0.8 + 0.75 * np.cos(rad)])
        V = _make_look_at_view(cam_pos, true_center)
        mvp = (P @ V)[None, :, :]
        proj = _project_corners_batch(gt_box.vertices[1:], mvp, np.array([w]), np.array([h]))[0]
        x1, y1 = float(np.min(proj[:, 0])), float(np.min(proj[:, 1]))
        x2, y2 = float(np.max(proj[:, 0])), float(np.max(proj[:, 1]))
        bboxes.append((x1, y1, x2, y2))
        ps.append(P)
        vs.append(V)
        ws.append(w)
        hs.append(h)

    pred_box, mean_iou2d = solve_multiview_cuboid_from_bboxes(bboxes, ps, vs, ws, hs)
    pred_dims = sorted(pred_box.dimensions_cm, reverse=True)
    gt_dims = sorted(gt_box.dimensions_cm, reverse=True)
    mae = float(np.mean(np.abs(np.array(pred_dims) - np.array(gt_dims))))

    assert mean_iou2d >= 0.90
    assert mae <= 1.5, f"Expected MAE <= 1.5 cm, got {mae:.2f} cm (pred={pred_dims}, gt={gt_dims})"
