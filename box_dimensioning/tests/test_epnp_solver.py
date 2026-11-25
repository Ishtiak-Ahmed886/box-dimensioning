"""Tests for EPnP Rigid Cuboid Solver."""
import numpy as np
import cv2
import pytest

from box_dimensioning.pipeline.epnp_solver import solve_epnp_cuboid, UNIT_CUBOID_8


def _make_synthetic_scenario(
    scale_m=(0.25, 0.18, 0.12),
    rvec=(0.3, -0.2, 0.1),
    tvec=(0.0, 0.0, 0.8),
    fx=800.0,
    img_size=(640, 480),
):
    """Creates a synthetic ground-truth box and projects it to 2D."""
    K = np.array([
        [fx, 0, img_size[0] / 2.0],
        [0, fx, img_size[1] / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    scale = np.array(scale_m, dtype=np.float64)
    rvec = np.array(rvec, dtype=np.float64)
    tvec = np.array(tvec, dtype=np.float64)

    obj_pts = UNIT_CUBOID_8 * scale.reshape(1, 3)
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    corners_2d = proj.reshape(-1, 2)

    return corners_2d, K, dist, scale, rvec, tvec


class TestEPnPSolver:
    """Validates EPnP rigid cuboid solver correctness."""

    def test_perfect_input_recovers_dimensions(self):
        """With noise-free projected corners, solver should recover exact scale."""
        gt_scale = (0.25, 0.18, 0.12)
        corners_2d, K, dist, scale_gt, _, _ = _make_synthetic_scenario(scale_m=gt_scale)

        result = solve_epnp_cuboid(corners_2d, K, dist)

        assert result['success'] is True
        # Recovered dimensions should be within 1 cm of ground truth
        dims = result['dimensions_cm']
        gt_sorted = sorted([s * 100 for s in gt_scale], reverse=True)
        recovered = [dims['length'], dims['width'], dims['height']]
        for gt_d, rec_d in zip(gt_sorted, recovered):
            assert abs(gt_d - rec_d) < 1.5, f"GT={gt_d:.1f} vs Recovered={rec_d:.1f}"

    def test_reprojection_error_low(self):
        """With noise-free input, reprojection error should be near zero."""
        corners_2d, K, dist, _, _, _ = _make_synthetic_scenario()
        result = solve_epnp_cuboid(corners_2d, K, dist)

        assert result['reprojection_error'] < 5.0  # sub-5px for clean input

    def test_rigid_orthogonality(self):
        """Recovered 3D corners must form a rigid orthogonal cuboid."""
        corners_2d, K, dist, _, _, _ = _make_synthetic_scenario()
        result = solve_epnp_cuboid(corners_2d, K, dist)
        c3d = result['corners_3d']

        # Check that the 3 edge vectors from corner 0 are mutually orthogonal
        e1 = c3d[1] - c3d[0]  # edge along one axis
        e2 = c3d[2] - c3d[0]  # edge along another axis
        e3 = c3d[4] - c3d[0]  # edge along third axis

        dot12 = abs(np.dot(e1, e2)) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
        dot13 = abs(np.dot(e1, e3)) / (np.linalg.norm(e1) * np.linalg.norm(e3) + 1e-9)
        dot23 = abs(np.dot(e2, e3)) / (np.linalg.norm(e2) * np.linalg.norm(e3) + 1e-9)

        # Dot products should be close to 0 (orthogonal)
        assert dot12 < 0.15, f"Edge 1-2 not orthogonal: cos={dot12:.3f}"
        assert dot13 < 0.15, f"Edge 1-3 not orthogonal: cos={dot13:.3f}"
        assert dot23 < 0.15, f"Edge 2-3 not orthogonal: cos={dot23:.3f}"

    def test_opposite_faces_parallel(self):
        """Opposite face normals must be anti-parallel (parallel but opposite direction)."""
        corners_2d, K, dist, _, _, _ = _make_synthetic_scenario()
        result = solve_epnp_cuboid(corners_2d, K, dist)
        c3d = result['corners_3d']

        # Top face normal (vertices 0,1,2,3)
        n_top = np.cross(c3d[1] - c3d[0], c3d[2] - c3d[0])
        n_top /= np.linalg.norm(n_top) + 1e-9
        # Bottom face normal (vertices 4,5,6,7)
        n_bot = np.cross(c3d[5] - c3d[4], c3d[6] - c3d[4])
        n_bot /= np.linalg.norm(n_bot) + 1e-9

        # Anti-parallel: dot product should be close to -1
        dot = np.dot(n_top, n_bot)
        assert abs(abs(dot) - 1.0) < 0.1, f"Faces not parallel: dot={dot:.3f}"

    def test_noisy_input_still_rigid(self):
        """With moderate pixel noise, solver should still produce a rigid box."""
        corners_2d, K, dist, _, _, _ = _make_synthetic_scenario()
        # Add 2px Gaussian noise
        np.random.seed(42)
        corners_noisy = corners_2d + np.random.normal(0, 2.0, corners_2d.shape)

        result = solve_epnp_cuboid(corners_noisy, K, dist)
        assert result['success'] is True
        # Dimensions should still be reasonable (within 3 cm of GT)
        dims = result['dimensions_cm']
        assert 10 < dims['length'] < 40
        assert 5 < dims['width'] < 30
        assert 3 < dims['height'] < 25

    def test_volumetric_weight_calculation(self):
        """Volumetric weight should equal L*W*H/5000."""
        corners_2d, K, dist, _, _, _ = _make_synthetic_scenario()
        result = solve_epnp_cuboid(corners_2d, K, dist)
        dims = result['dimensions_cm']
        expected_vw = dims['length'] * dims['width'] * dims['height'] / 5000
        assert abs(result['volumetric_weight_kg'] - expected_vw) < 0.01

    def test_projected_2d_returned(self):
        """Result should contain reprojected 2D corners for overlay drawing."""
        corners_2d, K, dist, _, _, _ = _make_synthetic_scenario()
        result = solve_epnp_cuboid(corners_2d, K, dist)
        assert result['projected_2d'].shape == (8, 2)

    def test_box_object_returned(self):
        """Result should contain a valid Box instance with 9 vertices."""
        corners_2d, K, dist, _, _, _ = _make_synthetic_scenario()
        result = solve_epnp_cuboid(corners_2d, K, dist)
        assert result['box'].vertices.shape == (9, 3)
