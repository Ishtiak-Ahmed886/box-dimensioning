"""OpenCV Camera Calibration module using chessboard pattern (F1, A1).
Meets acceptance criterion A1: RMS reprojection error <= 0.6 px on the OpenCV sample set.
"""

from typing import Tuple, List, Dict, Optional
import os
import glob
import json
import urllib.request
import cv2
import numpy as np


OPENCV_SAMPLES_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data"
DEFAULT_CHESSBOARD_IMAGES = [f"left{i:02d}.jpg" for i in range(1, 15)]  # left01.jpg to left14.jpg
DEFAULT_PATTERN_SIZE = (9, 6)  # 9 columns, 6 rows of internal corners
DEFAULT_SQUARE_SIZE_MM = 25.0  # standard 25mm chessboard squares


def download_opencv_chessboards(target_dir: str) -> List[str]:
    """Downloads left01.jpg - left14.jpg from OpenCV GitHub repository if not already present."""
    os.makedirs(target_dir, exist_ok=True)
    downloaded_paths = []

    for img_name in DEFAULT_CHESSBOARD_IMAGES:
        dest_path = os.path.join(target_dir, img_name)
        if not os.path.isfile(dest_path) or os.path.getsize(dest_path) == 0:
            url = f"{OPENCV_SAMPLES_URL}/{img_name}"
            try:
                urllib.request.urlretrieve(url, dest_path)
            except Exception as e:
                # Fallback: if network fails or sandbox blocks, log warning
                print(f"Warning: Could not download {url}: {e}")
        if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0:
            downloaded_paths.append(dest_path)

    return sorted(downloaded_paths)


class CameraCalibrator:
    """Calibrates camera intrinsics and lens distortion from chessboard image set."""

    def __init__(self, pattern_size: Tuple[int, int] = DEFAULT_PATTERN_SIZE, square_size_mm: float = DEFAULT_SQUARE_SIZE_MM):
        self.pattern_size = pattern_size
        self.square_size_mm = square_size_mm
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.rvecs: List[np.ndarray] = []
        self.tvecs: List[np.ndarray] = []
        self.rms_error: float = 0.0
        self.image_size: Tuple[int, int] = (0, 0)
        self.per_image_errors: Dict[str, float] = {}

    def calibrate(self, image_paths: List[str]) -> Dict:
        """Calibrates camera from list of chessboard image paths."""
        if not image_paths:
            raise ValueError("No images provided for calibration")

        # Prepare 3D object points in world coordinates: (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size_mm

        obj_points = []  # 3d point in real world space
        img_points = []  # 2d points in image plane
        valid_image_names = []

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.image_size = (gray.shape[1], gray.shape[0])

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )

            if ret and corners is not None:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
                obj_points.append(objp)
                img_points.append(corners_refined)
                valid_image_names.append(os.path.basename(img_path))

        if len(obj_points) < 3:
            raise RuntimeError(f"Found corners in only {len(obj_points)} images. At least 3 required.")

        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, self.image_size, None, None
        )

        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

        # Compute reprojection error for each image and total RMS error
        total_sq_error = 0.0
        total_points = 0
        self.per_image_errors = {}

        for i in range(len(obj_points)):
            img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
            pts_orig = img_points[i].reshape(-1, 2).astype(np.float64)
            pts_proj = img_points2.reshape(-1, 2).astype(np.float64)
            diff = pts_orig - pts_proj
            sq_err = np.sum(diff ** 2)
            num_pts = len(pts_orig)
            img_rmse = np.sqrt(sq_err / num_pts)
            self.per_image_errors[valid_image_names[i]] = float(img_rmse)
            total_sq_error += sq_err
            total_points += num_pts

        self.rms_error = float(np.sqrt(total_sq_error / total_points))

        return {
            "rms_reprojection_error_px": self.rms_error,
            "meets_acceptance_criterion_A1": self.rms_error <= 0.6,
            "image_size": list(self.image_size),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "valid_images_count": len(valid_image_names),
            "per_image_errors": self.per_image_errors,
        }

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """Undistorts an input image using computed camera matrix and distortion coefficients."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Camera is not calibrated yet")
        return cv2.undistort(img, self.camera_matrix, self.dist_coeffs)

    def save_calibration(self, output_json_path: str) -> None:
        """Saves calibration results to JSON."""
        data = {
            "rms_error": self.rms_error,
            "meets_A1": self.rms_error <= 0.6,
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else [],
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else [],
            "image_size": list(self.image_size),
            "per_image_errors": self.per_image_errors,
        }
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
