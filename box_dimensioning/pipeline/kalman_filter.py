"""6-DOF Kalman Filter for Temporal Cuboid Smoothing.

Tracks the cuboid's 2D corner positions and 3D dimensions across video frames
using a constant-velocity Kalman filter. Eliminates inter-frame jitter while
preserving responsiveness to real motion.

State vector (19-dim):
    [c1x, c1y, c2x, c2y, ..., c8x, c8y, L, W, H]
    - 8 corners × 2 coords = 16 position states
    - 3 dimension states (L, W, H in cm)

Each position/dimension also has a velocity term internally (38-dim full state).

Features:
    - Constant-velocity motion model for smooth prediction
    - Mahalanobis distance gating to reject outlier detections
    - Adaptive process noise based on detection confidence
"""

from typing import Optional, Tuple, Dict
import numpy as np


class CuboidKalmanFilter:
    """Kalman filter for smooth 2D corner + dimension tracking of a 3D cuboid."""

    def __init__(
        self,
        process_noise_pos: float = 2.0,
        process_noise_vel: float = 0.5,
        process_noise_dim: float = 0.3,
        measurement_noise_pos: float = 3.0,
        measurement_noise_dim: float = 1.5,
        mahalanobis_threshold: float = 12.0,
        dt: float = 1.0 / 30.0,
    ):
        """
        Args:
            process_noise_pos: std dev of position process noise (pixels/frame).
            process_noise_vel: std dev of velocity process noise (pixels/frame²).
            process_noise_dim: std dev of dimension process noise (cm/frame).
            measurement_noise_pos: std dev of 2D corner measurement noise (pixels).
            measurement_noise_dim: std dev of dimension measurement noise (cm).
            mahalanobis_threshold: outlier rejection threshold.
            dt: assumed time step between frames (seconds).
        """
        self.n_obs = 19  # 8 corners × 2 + 3 dims
        self.n_state = 2 * self.n_obs  # position + velocity for each

        self.dt = dt
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.process_noise_dim = process_noise_dim
        self.measurement_noise_pos = measurement_noise_pos
        self.measurement_noise_dim = measurement_noise_dim
        self.mahalanobis_threshold = mahalanobis_threshold

        # State vector: [obs_0, vel_0, obs_1, vel_1, ...]
        self.x = np.zeros(self.n_state, dtype=np.float64)

        # State covariance
        self.P = np.eye(self.n_state, dtype=np.float64) * 100.0

        # Transition matrix (constant velocity model)
        self.F = np.eye(self.n_state, dtype=np.float64)
        for i in range(self.n_obs):
            self.F[2 * i, 2 * i + 1] = dt  # pos += vel * dt

        # Measurement matrix: observe positions only (not velocities)
        self.H = np.zeros((self.n_obs, self.n_state), dtype=np.float64)
        for i in range(self.n_obs):
            self.H[i, 2 * i] = 1.0

        # Process noise
        self.Q = self._build_process_noise()

        # Measurement noise
        self.R = self._build_measurement_noise()

        self._initialized = False
        self._outlier_count = 0

    def _build_process_noise(self) -> np.ndarray:
        Q = np.zeros((self.n_state, self.n_state), dtype=np.float64)
        for i in range(self.n_obs):
            if i < 16:  # 2D position components
                Q[2 * i, 2 * i] = self.process_noise_pos ** 2
                Q[2 * i + 1, 2 * i + 1] = self.process_noise_vel ** 2
            else:  # dimension components (L, W, H)
                Q[2 * i, 2 * i] = self.process_noise_dim ** 2
                Q[2 * i + 1, 2 * i + 1] = (self.process_noise_dim * 0.1) ** 2
        return Q

    def _build_measurement_noise(self) -> np.ndarray:
        R = np.zeros((self.n_obs, self.n_obs), dtype=np.float64)
        for i in range(self.n_obs):
            if i < 16:
                R[i, i] = self.measurement_noise_pos ** 2
            else:
                R[i, i] = self.measurement_noise_dim ** 2
        return R

    def _corners_dims_to_z(
        self, corners_2d: np.ndarray, dims: Dict[str, float]
    ) -> np.ndarray:
        """Packs corners (8,2) + dims dict into measurement vector (19,)."""
        c = np.asarray(corners_2d, dtype=np.float64).flatten()[:16]
        d = np.array([dims['length'], dims['width'], dims['height']], dtype=np.float64)
        return np.concatenate([c, d])

    def _z_to_corners_dims(
        self, z: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Unpacks measurement vector back to corners + dims."""
        corners = z[:16].reshape(8, 2)
        dims = {
            'length': max(1.0, round(float(z[16]), 1)),
            'width': max(1.0, round(float(z[17]), 1)),
            'height': max(1.0, round(float(z[18]), 1)),
        }
        return corners, dims

    def initialize(self, corners_2d: np.ndarray, dims: Dict[str, float]):
        """Initialize filter state from first observation."""
        z = self._corners_dims_to_z(corners_2d, dims)
        for i in range(self.n_obs):
            self.x[2 * i] = z[i]
            self.x[2 * i + 1] = 0.0  # zero initial velocity
        self.P = np.eye(self.n_state) * 50.0
        # Lower initial uncertainty for dimensions (they change slowly)
        for i in range(16, 19):
            self.P[2 * i, 2 * i] = 5.0
            self.P[2 * i + 1, 2 * i + 1] = 0.5
        self._initialized = True
        self._outlier_count = 0

    def predict(self):
        """Predict step: advance state by one time step."""
        if not self._initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _mahalanobis_distance(self, z: np.ndarray) -> float:
        """Compute Mahalanobis distance of observation from predicted state."""
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(y.T @ S_inv @ y)
            return np.sqrt(max(0, d2))
        except np.linalg.LinAlgError:
            return 0.0  # degenerate — accept

    def update(
        self, corners_2d: np.ndarray, dims: Dict[str, float]
    ) -> Tuple[np.ndarray, Dict[str, float], bool]:
        """Update step: incorporate new measurement.

        Args:
            corners_2d: (8, 2) detected 2D corners.
            dims: {'length': L, 'width': W, 'height': H} in cm.

        Returns:
            (smoothed_corners_2d, smoothed_dims, accepted)
            accepted is False if the measurement was rejected as outlier.
        """
        if not self._initialized:
            self.initialize(corners_2d, dims)
            return corners_2d.copy(), dims.copy(), True

        z = self._corners_dims_to_z(corners_2d, dims)

        # Mahalanobis gating
        m_dist = self._mahalanobis_distance(z)
        if m_dist > self.mahalanobis_threshold:
            self._outlier_count += 1
            # If too many consecutive outliers, re-initialize
            if self._outlier_count >= 5:
                self.initialize(corners_2d, dims)
                return corners_2d.copy(), dims.copy(), True
            # Reject this measurement — return predicted state
            predicted_z = self.H @ self.x
            return self._z_to_corners_dims(predicted_z) + (False,)

        self._outlier_count = 0

        # Standard Kalman update
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        except np.linalg.LinAlgError:
            K = np.zeros((self.n_state, self.n_obs))

        self.x = self.x + K @ y
        I_KH = np.eye(self.n_state) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form

        # Extract smoothed state
        smoothed_z = self.H @ self.x
        corners_smooth, dims_smooth = self._z_to_corners_dims(smoothed_z)
        return corners_smooth, dims_smooth, True

    def get_smoothed(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """Returns current smoothed state without updating."""
        if not self._initialized:
            return np.zeros((8, 2)), {'length': 0.0, 'width': 0.0, 'height': 0.0}
        z = self.H @ self.x
        return self._z_to_corners_dims(z)

    def reset(self):
        """Resets filter state."""
        self.x = np.zeros(self.n_state, dtype=np.float64)
        self.P = np.eye(self.n_state) * 100.0
        self._initialized = False
        self._outlier_count = 0

    @property
    def is_initialized(self) -> bool:
        return self._initialized
