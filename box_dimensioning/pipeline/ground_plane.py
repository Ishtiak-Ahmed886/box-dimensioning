"""Ground Plane Estimation and Cuboid Snapping.

Estimates the support surface (table, floor) from AR point clouds or heuristics,
then snaps a 3D cuboid's base flush onto that plane — eliminating floating or
tilted boxes.

Sources (in priority order):
  1. AR metadata plane anchors (PlaneInfo from Objectron protobuf)
  2. RANSAC plane fitting on sparse 3D point cloud
  3. Heuristic: assume a horizontal surface (gravity = -Y in camera frame)
"""

from typing import Optional, Tuple
import numpy as np


def estimate_plane_ransac(
    points: np.ndarray,
    n_iterations: int = 200,
    distance_threshold: float = 0.015,
    min_inliers_ratio: float = 0.3,
) -> Optional[Tuple[np.ndarray, float]]:
    """Fits a plane to a 3D point cloud using RANSAC.

    Args:
        points: (N, 3) array of 3D points.
        n_iterations: number of RANSAC iterations.
        distance_threshold: inlier distance in meters.
        min_inliers_ratio: minimum fraction of inliers to accept the plane.

    Returns:
        (normal, d) where normal is unit normal (3,) and d is the signed offset
        such that normal · x + d = 0 for points on the plane.
        Returns None if no dominant plane found.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None

    n_pts = len(points)
    best_inliers = 0
    best_normal = None
    best_d = 0.0

    rng = np.random.RandomState(42)
    for _ in range(n_iterations):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p0)

        distances = np.abs(points @ normal + d)
        n_inliers = int(np.sum(distances < distance_threshold))

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal.copy()
            best_d = d

    if best_normal is None or best_inliers < n_pts * min_inliers_ratio:
        return None

    # Refine with all inliers via least-squares
    distances = np.abs(points @ best_normal + best_d)
    inlier_mask = distances < distance_threshold
    inlier_pts = points[inlier_mask]
    if len(inlier_pts) >= 3:
        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        refined_normal = vh[-1]
        if np.dot(refined_normal, best_normal) < 0:
            refined_normal = -refined_normal
        best_normal = refined_normal / np.linalg.norm(refined_normal)
        best_d = -np.dot(best_normal, centroid)

    return best_normal, best_d


def get_ground_normal(
    ar_planes: Optional[list] = None,
    point_cloud: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Returns the best-estimate ground plane normal (pointing up/away from surface).

    Priority:
      1. AR metadata horizontal plane anchor
      2. RANSAC on point cloud
      3. Heuristic: Y-up in camera frame (standard for phone held at ~30° over table)

    Returns:
        (3,) unit normal vector pointing away from the support surface.
    """
    # Priority 1: AR horizontal plane anchor
    if ar_planes:
        for plane in ar_planes:
            if hasattr(plane, 'alignment') and plane.alignment == 1:  # HORIZONTAL
                n = np.asarray(plane.normal, dtype=np.float64)
                n_len = np.linalg.norm(n)
                if n_len > 1e-6:
                    n = n / n_len
                    # Ensure normal points "up" (positive Y component in most coordinate systems)
                    if n[1] < 0:
                        n = -n
                    return n

        # Any plane if no horizontal found
        n = np.asarray(ar_planes[0].normal, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len > 1e-6:
            n = n / n_len
            if n[1] < 0:
                n = -n
            return n

    # Priority 2: RANSAC on point cloud
    if point_cloud is not None and len(point_cloud) >= 10:
        result = estimate_plane_ransac(point_cloud)
        if result is not None:
            normal, _ = result
            if normal[1] < 0:
                normal = -normal
            return normal

    # Priority 3: Heuristic — assume Y-up
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def snap_cuboid_to_ground(
    corners_3d: np.ndarray,
    ground_normal: np.ndarray,
    ground_d: Optional[float] = None,
) -> np.ndarray:
    """Snaps a cuboid's base face flush onto the ground plane.

    Finds the 4 corners closest to the ground plane and translates the entire
    cuboid so those 4 corners lie exactly on the plane.

    Args:
        corners_3d: (8, 3) cuboid corner positions.
        ground_normal: (3,) unit normal of ground plane (pointing up).
        ground_d: plane offset such that n·x + d = 0. If None, estimated from
                  the lowest 4 corners.

    Returns:
        (8, 3) adjusted corner positions with base flush on ground.
    """
    corners = np.asarray(corners_3d, dtype=np.float64).copy()
    normal = np.asarray(ground_normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # Signed distances from each corner to ground plane
    if ground_d is not None:
        signed_dists = corners @ normal + ground_d
    else:
        # Estimate plane from 4 lowest corners
        projections = corners @ normal
        sorted_idx = np.argsort(projections)
        base_indices = sorted_idx[:4]
        base_center = np.mean(corners[base_indices], axis=0)
        ground_d = -np.dot(normal, base_center)
        signed_dists = corners @ normal + ground_d

    # Find the 4 base corners (closest to ground)
    abs_dists = np.abs(signed_dists)
    base_idx = np.argsort(abs_dists)[:4]

    # Compute mean offset of base corners from ground
    base_offset = float(np.mean(signed_dists[base_idx]))

    # Translate entire cuboid so base sits on ground
    corners -= base_offset * normal.reshape(1, 3)

    return corners


def align_rotation_to_gravity(
    R: np.ndarray,
    gravity_normal: np.ndarray,
    gravity_axis: int = 1,
) -> np.ndarray:
    """Adjusts rotation matrix so one axis aligns with gravity direction.

    Finds the rotation column closest to gravity_normal and rotates the box
    so that column aligns exactly with gravity, while minimizing change to
    the other two axes (preserving the box's yaw on the table).

    Args:
        R: (3, 3) rotation matrix of the cuboid.
        gravity_normal: (3,) unit gravity direction (up).
        gravity_axis: which column of R should align with gravity (default 1 = Y-up).

    Returns:
        (3, 3) corrected rotation matrix.
    """
    R = np.asarray(R, dtype=np.float64).copy()
    g = np.asarray(gravity_normal, dtype=np.float64)
    g = g / (np.linalg.norm(g) + 1e-12)

    # Find which existing axis is closest to gravity
    dots = np.abs(R.T @ g)
    closest_axis = int(np.argmax(dots))

    # Swap if needed so gravity_axis aligns
    if closest_axis != gravity_axis:
        R[:, [gravity_axis, closest_axis]] = R[:, [closest_axis, gravity_axis]]

    # Ensure correct sign
    if np.dot(R[:, gravity_axis], g) < 0:
        R[:, gravity_axis] = -R[:, gravity_axis]

    # Now force exact alignment: set the gravity column to g
    R[:, gravity_axis] = g

    # Re-orthogonalize via Gram-Schmidt on the other two axes
    axes = [i for i in range(3) if i != gravity_axis]
    a0, a1 = axes

    # Remove gravity component from first non-gravity axis
    R[:, a0] = R[:, a0] - np.dot(R[:, a0], g) * g
    n0 = np.linalg.norm(R[:, a0])
    if n0 > 1e-9:
        R[:, a0] /= n0
    else:
        # Degenerate: pick arbitrary perpendicular
        R[:, a0] = np.cross(g, np.array([1, 0, 0]) if abs(g[0]) < 0.9 else np.array([0, 1, 0]))
        R[:, a0] /= np.linalg.norm(R[:, a0])

    # Third axis = cross product to ensure right-handedness
    R[:, a1] = np.cross(R[:, gravity_axis], R[:, a0])
    R[:, a1] /= np.linalg.norm(R[:, a1]) + 1e-12

    # Ensure det(R) = +1
    if np.linalg.det(R) < 0:
        R[:, a1] = -R[:, a1]

    return R
