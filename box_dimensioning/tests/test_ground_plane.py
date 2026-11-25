"""Tests for Ground Plane Estimation and Cuboid Snapping."""
import numpy as np
import pytest

from box_dimensioning.pipeline.ground_plane import (
    estimate_plane_ransac,
    get_ground_normal,
    snap_cuboid_to_ground,
    align_rotation_to_gravity,
)


class TestRANSACPlane:
    def test_perfect_horizontal_plane(self):
        """Points on y=0.5 plane should yield normal ≈ [0, 1, 0]."""
        rng = np.random.RandomState(0)
        pts = np.column_stack([
            rng.uniform(-1, 1, 50),
            np.full(50, 0.5) + rng.normal(0, 0.002, 50),
            rng.uniform(-1, 1, 50),
        ])
        result = estimate_plane_ransac(pts)
        assert result is not None
        normal, d = result
        # Normal should be close to [0, ±1, 0]
        assert abs(abs(normal[1]) - 1.0) < 0.05

    def test_too_few_points_returns_none(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]])
        assert estimate_plane_ransac(pts) is None

    def test_noisy_tilted_plane(self):
        """Should find plane even with noise and tilt."""
        rng = np.random.RandomState(7)
        n = 80
        u = rng.uniform(-1, 1, n)
        v = rng.uniform(-1, 1, n)
        # Tilted plane: z = 0.3*x + 0.2*y + 1.0
        z = 0.3 * u + 0.2 * v + 1.0 + rng.normal(0, 0.005, n)
        pts = np.column_stack([u, v, z])
        result = estimate_plane_ransac(pts)
        assert result is not None


class TestGetGroundNormal:
    def test_heuristic_fallback(self):
        """With no data, should return Y-up."""
        n = get_ground_normal()
        np.testing.assert_allclose(n, [0, 1, 0])

    def test_ar_plane_priority(self):
        """AR horizontal plane should take priority."""
        class FakePlane:
            alignment = 1  # HORIZONTAL
            normal = np.array([0.0, 0.98, 0.2])
        n = get_ground_normal(ar_planes=[FakePlane()])
        assert abs(np.linalg.norm(n) - 1.0) < 1e-6
        assert n[1] > 0.9  # mostly up


class TestSnapCuboid:
    def test_floating_box_snaps_down(self):
        """A box floating 10 cm above ground should snap to y=0."""
        ground_normal = np.array([0, 1, 0], dtype=np.float64)
        # Box centered at y=0.15, half-height=0.05 → base at y=0.10 (floating)
        corners = np.array([
            [-0.1, 0.10, -0.1],
            [+0.1, 0.10, -0.1],
            [-0.1, 0.20, -0.1],
            [+0.1, 0.20, -0.1],
            [-0.1, 0.10, +0.1],
            [+0.1, 0.10, +0.1],
            [-0.1, 0.20, +0.1],
            [+0.1, 0.20, +0.1],
        ], dtype=np.float64)
        snapped = snap_cuboid_to_ground(corners, ground_normal, ground_d=0.0)
        # Base corners (y-min = indices 0,1,4,5) should now be at y≈0
        base_y = snapped[[0, 1, 4, 5], 1]
        assert np.all(np.abs(base_y) < 0.02), f"Base Y values: {base_y}"

    def test_shape_preserved(self):
        """Snapping should not change box dimensions."""
        ground_normal = np.array([0, 1, 0])
        corners = np.array([
            [0, 0.5, 0], [1, 0.5, 0], [0, 1.5, 0], [1, 1.5, 0],
            [0, 0.5, 1], [1, 0.5, 1], [0, 1.5, 1], [1, 1.5, 1],
        ], dtype=np.float64)
        snapped = snap_cuboid_to_ground(corners, ground_normal)
        # Edge lengths should be preserved
        orig_edge = np.linalg.norm(corners[1] - corners[0])
        snap_edge = np.linalg.norm(snapped[1] - snapped[0])
        assert abs(orig_edge - snap_edge) < 1e-10


class TestAlignRotation:
    def test_identity_stays_identity(self):
        """If box is already gravity-aligned, rotation should barely change."""
        R = np.eye(3)
        gravity = np.array([0, 1, 0])
        R_aligned = align_rotation_to_gravity(R, gravity, gravity_axis=1)
        assert abs(np.linalg.det(R_aligned) - 1.0) < 1e-6
        np.testing.assert_allclose(R_aligned[:, 1], gravity, atol=1e-10)

    def test_tilted_box_aligns(self):
        """A 30° tilted box should be corrected to upright."""
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        gravity = np.array([0, 1, 0])
        R_aligned = align_rotation_to_gravity(R, gravity, gravity_axis=1)
        # Y column should now be exactly [0, 1, 0]
        np.testing.assert_allclose(R_aligned[:, 1], gravity, atol=1e-10)
        # Must remain valid rotation
        assert abs(np.linalg.det(R_aligned) - 1.0) < 1e-6
        # Orthogonality check
        RtR = R_aligned.T @ R_aligned
        np.testing.assert_allclose(RtR, np.eye(3), atol=1e-9)
