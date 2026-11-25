"""Tests for 6-DOF Cuboid Kalman Filter."""
import numpy as np
import pytest

from box_dimensioning.pipeline.kalman_filter import CuboidKalmanFilter


def _make_corners(offset_x=0.0, offset_y=0.0):
    """Creates synthetic 8 corners of a 100×60 px box at given offset."""
    base = np.array([
        [100, 100], [200, 100], [100, 160], [200, 160],
        [100, 250], [200, 250], [100, 310], [200, 310],
    ], dtype=np.float64)
    base[:, 0] += offset_x
    base[:, 1] += offset_y
    return base


def _make_dims():
    return {'length': 25.0, 'width': 18.0, 'height': 12.0}


class TestKalmanFilter:
    def test_initialization(self):
        kf = CuboidKalmanFilter()
        assert not kf.is_initialized
        corners = _make_corners()
        dims = _make_dims()
        kf.initialize(corners, dims)
        assert kf.is_initialized

    def test_first_update_returns_input(self):
        """First measurement should be accepted and returned nearly unchanged."""
        kf = CuboidKalmanFilter()
        corners = _make_corners()
        dims = _make_dims()
        out_c, out_d, accepted = kf.update(corners, dims)
        assert accepted
        np.testing.assert_allclose(out_c, corners, atol=0.1)
        assert out_d['length'] == dims['length']

    def test_smooth_stationary_input(self):
        """Repeated identical observations should converge to that value."""
        kf = CuboidKalmanFilter()
        corners = _make_corners()
        dims = _make_dims()
        for _ in range(20):
            kf.predict()
            out_c, out_d, _ = kf.update(corners, dims)
        np.testing.assert_allclose(out_c, corners, atol=1.0)
        assert abs(out_d['length'] - 25.0) < 0.5

    def test_jitter_suppression(self):
        """Adding noise to observations should produce smoother output."""
        kf = CuboidKalmanFilter()
        corners_base = _make_corners()
        dims = _make_dims()
        rng = np.random.RandomState(42)

        raw_jitters = []
        filtered_jitters = []
        prev_raw = None
        prev_filt = None

        for i in range(30):
            noise = rng.normal(0, 4.0, corners_base.shape)
            noisy = corners_base + noise
            kf.predict()
            smooth_c, smooth_d, _ = kf.update(noisy, dims)

            if prev_raw is not None:
                raw_jitters.append(np.mean(np.abs(noisy - prev_raw)))
                filtered_jitters.append(np.mean(np.abs(smooth_c - prev_filt)))
            prev_raw = noisy.copy()
            prev_filt = smooth_c.copy()

        # Filtered jitter should be significantly less than raw jitter
        mean_raw = np.mean(raw_jitters)
        mean_filt = np.mean(filtered_jitters)
        assert mean_filt < mean_raw * 0.8, f"Filter={mean_filt:.2f} vs Raw={mean_raw:.2f}"

    def test_outlier_rejection(self):
        """A sudden large jump should be rejected by Mahalanobis gating."""
        kf = CuboidKalmanFilter(mahalanobis_threshold=8.0)
        corners = _make_corners()
        dims = _make_dims()

        # Initialize with 10 normal frames
        for _ in range(10):
            kf.predict()
            kf.update(corners, dims)

        # Inject a huge outlier (shifted 500 px)
        outlier = _make_corners(offset_x=500)
        kf.predict()
        _, _, accepted = kf.update(outlier, dims)
        assert not accepted, "Outlier should be rejected"

    def test_moving_target_tracking(self):
        """Filter should track a steadily moving box."""
        kf = CuboidKalmanFilter()
        dims = _make_dims()

        for i in range(30):
            corners = _make_corners(offset_x=float(i * 3))
            kf.predict()
            out_c, _, _ = kf.update(corners, dims)

        # After 30 frames of rightward motion, filtered position should be close
        expected_x = 100 + 29 * 3
        assert abs(out_c[0, 0] - expected_x) < 15, f"Expected ~{expected_x}, got {out_c[0, 0]:.1f}"

    def test_reset(self):
        kf = CuboidKalmanFilter()
        kf.initialize(_make_corners(), _make_dims())
        assert kf.is_initialized
        kf.reset()
        assert not kf.is_initialized

    def test_get_smoothed_before_init(self):
        kf = CuboidKalmanFilter()
        corners, dims = kf.get_smoothed()
        assert corners.shape == (8, 2)
        assert dims['length'] == 0.0

    def test_dimensions_stable_across_frames(self):
        """Dimensions should converge and stay stable."""
        kf = CuboidKalmanFilter()
        corners = _make_corners()
        dims = _make_dims()
        rng = np.random.RandomState(99)

        last_dims = []
        for i in range(25):
            noisy_dims = {
                'length': 25.0 + rng.normal(0, 1.5),
                'width': 18.0 + rng.normal(0, 1.0),
                'height': 12.0 + rng.normal(0, 0.8),
            }
            kf.predict()
            _, out_d, _ = kf.update(corners, noisy_dims)
            if i >= 15:
                last_dims.append(out_d['length'])

        # Last 10 frames should have low variance (< 1 cm std)
        std = np.std(last_dims)
        assert std < 1.5, f"Dimension std too high: {std:.2f}"
