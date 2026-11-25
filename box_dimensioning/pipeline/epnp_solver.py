"""EPnP Rigid Cuboid Solver: Single-frame 3D box pose and dimension estimation.

Uses Perspective-n-Point (EPnP) with a parameterized cuboid model to recover
rigid 3D bounding box pose (R, t) and anisotropic scale (sx, sy, sz) from
8 detected 2D corner keypoints and camera intrinsics K.

Key insight: Unlike unconstrained DLT triangulation where each 3D point is
independent (causing depth stretch on thin boxes with low baselines), EPnP
enforces that all 8 corners belong to a single rigid orthogonal cuboid.
Opposite faces are guaranteed parallel and equal by construction.

References:
    - Lepetit, Moreno-Noguer, Fua. "EPnP: Efficient PnP." IJCV 2009.
    - Ahmadyan et al. "Objectron: 3D Object Detection." CVPR 2021.
    - OpenCV: cv2.solvePnP, cv2.solvePnPRefineLM
"""

from typing import Tuple, Dict, Optional
import numpy as np
import cv2
from scipy.optimize import minimize

from box_dimensioning.dataset.box_geometry import Box, EDGES, compute_volumetric_weight


# Canonical unit cuboid: 8 vertices of a 1×1×1 box centered at origin.
# Vertex ordering matches Objectron/box_geometry convention (indices 1-8):
#   Top face (y=+0.5): vertices 3,4,7,8
#   Bottom face (y=-0.5): vertices 1,2,5,6
#   Front face (z=+0.5): vertices 2,4,6,8
#   Back face (z=-0.5): vertices 1,3,5,7
UNIT_CUBOID_8 = np.array([
    [-0.5, -0.5, -0.5],  # vertex 1
    [-0.5, -0.5, +0.5],  # vertex 2
    [-0.5, +0.5, -0.5],  # vertex 3
    [-0.5, +0.5, +0.5],  # vertex 4
    [+0.5, -0.5, -0.5],  # vertex 5
    [+0.5, -0.5, +0.5],  # vertex 6
    [+0.5, +0.5, -0.5],  # vertex 7
    [+0.5, +0.5, +0.5],  # vertex 8
], dtype=np.float64)


def _reprojection_error(
    params: np.ndarray,
    corners_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """Cost function: total reprojection error for a parameterized rigid cuboid.

    params: [rx, ry, rz, tx, ty, tz, sx, sy, sz] — 9 parameters
        - (rx, ry, rz): Rodrigues rotation vector
        - (tx, ty, tz): translation in camera frame
        - (sx, sy, sz): anisotropic scale (box dimensions in meters)
    """
    rvec = params[:3]
    tvec = params[3:6]
    scale = np.abs(params[6:9])  # enforce positive scale
    scale = np.maximum(scale, 0.01)  # minimum 1 cm

    # Scale the unit cuboid to actual dimensions
    object_pts = UNIT_CUBOID_8 * scale.reshape(1, 3)

    # Project to 2D
    projected, _ = cv2.projectPoints(
        object_pts, rvec, tvec, K, dist_coeffs
    )
    projected = projected.reshape(-1, 2)

    # Sum of squared reprojection errors
    diff = corners_2d - projected
    return float(np.sum(diff ** 2))


def solve_epnp_cuboid(
    corners_2d: np.ndarray,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    initial_scale: Optional[np.ndarray] = None,
    ground_normal: Optional[np.ndarray] = None,
) -> Dict:
    """Estimates 3D rigid cuboid pose and dimensions from 8 detected 2D corners.

    Strategy:
    1. Use an initial scale guess (or default 20×15×10 cm) to create 3D model points.
    2. Run cv2.solvePnP (EPNP) to get initial pose (R, t).
    3. Refine pose with cv2.solvePnPRefineLM for sub-pixel accuracy.
    4. Run Nelder-Mead optimization over all 9 parameters (R, t, scale) to
       jointly minimize reprojection error while maintaining rigid cuboid.
    5. If ground_normal is provided, snap the cuboid base flush onto the ground
       plane and align the box's gravity axis.

    Args:
        corners_2d: (8, 2) array of detected 2D corner keypoints in image pixels.
        K: (3, 3) camera intrinsic matrix.
        dist_coeffs: distortion coefficients. Defaults to zero.
        initial_scale: (3,) initial guess for box dimensions [sx, sy, sz] in meters.
                       Defaults to [0.20, 0.15, 0.10] (a typical small parcel).
        ground_normal: (3,) unit normal of the support surface (pointing up).
                       If provided, the cuboid's Y-axis is aligned with this normal
                       and the base is snapped flush onto the ground plane.

    Returns:
        dict with keys:
            - 'corners_3d': (8, 3) corners in camera coordinate frame
            - 'center_3d': (3,) center of the cuboid
            - 'dimensions_cm': {'length': L, 'width': W, 'height': H}
            - 'volumetric_weight_kg': float
            - 'rvec': (3,) Rodrigues rotation
            - 'tvec': (3,) translation
            - 'scale_m': (3,) recovered scale in meters
            - 'reprojection_error': float (mean per-corner px error)
            - 'projected_2d': (8, 2) reprojected 2D corners (for overlay)
            - 'box': Box instance
            - 'success': bool
    """
    corners_2d = np.asarray(corners_2d, dtype=np.float64).reshape(8, 2)
    K = np.asarray(K, dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)

    if initial_scale is None:
        initial_scale = np.array([0.20, 0.15, 0.10])
    initial_scale = np.asarray(initial_scale, dtype=np.float64)

    # --- Stage 1: Initial PnP solve with current scale guess ---
    object_pts_init = (UNIT_CUBOID_8 * initial_scale.reshape(1, 3)).astype(np.float64)
    image_pts = corners_2d.reshape(-1, 1, 2).astype(np.float64)

    # Try multiple PnP methods for robustness
    best_rvec, best_tvec = None, None
    best_reproj = float('inf')

    for method in [cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_SQPNP]:
        try:
            success, rvec, tvec = cv2.solvePnP(
                object_pts_init, image_pts, K, dist_coeffs, flags=method
            )
            if not success:
                continue

            # Compute reprojection error for this method
            projected, _ = cv2.projectPoints(object_pts_init, rvec, tvec, K, dist_coeffs)
            err = float(np.mean(np.linalg.norm(
                corners_2d - projected.reshape(-1, 2), axis=1
            )))
            if err < best_reproj:
                best_reproj = err
                best_rvec = rvec.flatten()
                best_tvec = tvec.flatten()
        except cv2.error:
            continue

    if best_rvec is None:
        # Absolute fallback: use naive back-projection
        return _fallback_backproject(corners_2d, K, initial_scale)

    # --- Stage 2: Levenberg-Marquardt pose refinement ---
    try:
        rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
            object_pts_init, image_pts, K, dist_coeffs,
            best_rvec.reshape(3, 1), best_tvec.reshape(3, 1)
        )
        best_rvec = rvec_ref.flatten()
        best_tvec = tvec_ref.flatten()
    except cv2.error:
        pass  # keep initial PnP result

    # --- Stage 3: Joint optimization of pose + scale ---
    x0 = np.concatenate([best_rvec, best_tvec, initial_scale])

    result = minimize(
        _reprojection_error,
        x0,
        args=(corners_2d, K, dist_coeffs),
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-3, 'adaptive': True}
    )

    if result.success or result.fun < best_reproj * 8:
        opt_params = result.x
    else:
        opt_params = x0  # stick with PnP result if optimization diverged

    final_rvec = opt_params[:3]
    final_tvec = opt_params[3:6]
    final_scale = np.abs(opt_params[6:9])
    final_scale = np.maximum(final_scale, 0.01)

    # --- Stage 4: Reconstruct final 3D cuboid corners in camera frame ---
    R, _ = cv2.Rodrigues(final_rvec)
    object_pts_scaled = UNIT_CUBOID_8 * final_scale.reshape(1, 3)

    # --- Stage 4b: Ground plane alignment and snapping ---
    if ground_normal is not None:
        from box_dimensioning.pipeline.ground_plane import (
            align_rotation_to_gravity, snap_cuboid_to_ground
        )
        gn = np.asarray(ground_normal, dtype=np.float64)
        gn = gn / (np.linalg.norm(gn) + 1e-12)
        R = align_rotation_to_gravity(R, gn, gravity_axis=1)

    corners_3d_cam = (R @ object_pts_scaled.T).T + final_tvec.reshape(1, 3)

    if ground_normal is not None:
        corners_3d_cam = snap_cuboid_to_ground(corners_3d_cam, gn)

    center_3d = np.mean(corners_3d_cam, axis=0)

    # Compute final reprojection
    projected_final, _ = cv2.projectPoints(
        object_pts_scaled, final_rvec, final_tvec, K, dist_coeffs
    )
    projected_final = projected_final.reshape(-1, 2)
    reproj_error = float(np.mean(np.linalg.norm(
        corners_2d - projected_final, axis=1
    )))

    # Extract dimensions in cm, ordered L >= W >= H
    dims_cm = sorted([
        round(float(final_scale[0]) * 100, 1),
        round(float(final_scale[1]) * 100, 1),
        round(float(final_scale[2]) * 100, 1),
    ], reverse=True)
    dims_cm = [max(1.0, d) for d in dims_cm]

    dimensions = {
        "length": dims_cm[0],
        "width": dims_cm[1],
        "height": dims_cm[2],
    }
    vol_wt = compute_volumetric_weight(dims_cm[0], dims_cm[1], dims_cm[2])

    # Build Box object (9 vertices: center + 8 corners)
    vertices_9 = np.vstack([center_3d.reshape(1, 3), corners_3d_cam])
    box = Box(vertices=vertices_9)

    return {
        'corners_3d': corners_3d_cam,
        'center_3d': center_3d,
        'dimensions_cm': dimensions,
        'volumetric_weight_kg': round(vol_wt, 4),
        'rvec': final_rvec,
        'tvec': final_tvec,
        'scale_m': final_scale,
        'reprojection_error': round(reproj_error, 2),
        'projected_2d': projected_final,
        'box': box,
        'success': True,
    }


def _fallback_backproject(
    corners_2d: np.ndarray,
    K: np.ndarray,
    scale: np.ndarray,
) -> Dict:
    """Fallback: naive back-projection when PnP fails entirely."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Estimate depth from apparent size vs expected scale
    bbox_w = np.max(corners_2d[:, 0]) - np.min(corners_2d[:, 0])
    bbox_h = np.max(corners_2d[:, 1]) - np.min(corners_2d[:, 1])
    expected_size_m = max(scale[0], scale[1])
    apparent_size_px = max(bbox_w, bbox_h, 1.0)
    z_est = expected_size_m * fx / apparent_size_px

    corners_3d = np.zeros((8, 3), dtype=np.float64)
    for i in range(8):
        u, v = corners_2d[i]
        corners_3d[i, 0] = (u - cx) * z_est / fx
        corners_3d[i, 1] = (v - cy) * z_est / fy
        corners_3d[i, 2] = z_est

    center_3d = np.mean(corners_3d, axis=0)
    dims_m = [
        float(np.linalg.norm(corners_3d[1] - corners_3d[0])),
        float(np.linalg.norm(corners_3d[2] - corners_3d[0])),
        float(np.linalg.norm(corners_3d[4] - corners_3d[0])),
    ]
    dims_cm = sorted([max(1.0, round(d * 100, 1)) for d in dims_m], reverse=True)
    dimensions = {"length": dims_cm[0], "width": dims_cm[1], "height": dims_cm[2]}
    vol_wt = compute_volumetric_weight(dims_cm[0], dims_cm[1], dims_cm[2])

    vertices_9 = np.vstack([center_3d.reshape(1, 3), corners_3d])
    box = Box(vertices=vertices_9)

    return {
        'corners_3d': corners_3d,
        'center_3d': center_3d,
        'dimensions_cm': dimensions,
        'volumetric_weight_kg': round(vol_wt, 4),
        'rvec': np.zeros(3),
        'tvec': np.array([0, 0, z_est]),
        'scale_m': np.array(dims_m),
        'reprojection_error': 999.0,
        'projected_2d': corners_2d.copy(),
        'box': box,
        'success': False,
    }
