"""General 3D Bounding Box and 3D IoU geometry utilities.
Based on and compatible with the Google Research Objectron dataset geometry specifications.
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as rotation_util
import scipy.spatial as sp

# Box Edges connecting the 8 bounding box vertices (1-indexed matching 9 keypoints with 0 as center)
EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

# Faces ordered such that normal points inward according to left-hand rule
FACES = np.array([
    [5, 6, 8, 7],  # +x on yz plane
    [1, 3, 4, 2],  # -x on yz plane
    [3, 7, 8, 4],  # +y on xz plane = top
    [1, 2, 6, 5],  # -y on xz plane
    [2, 4, 8, 6],  # +z on xy plane = front
    [1, 5, 7, 3],  # -z on xy plane
])

UNIT_BOX = np.asarray([
    [0.0, 0.0, 0.0],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
])

NUM_KEYPOINTS = 9
_PLANE_THICKNESS_EPSILON = 1e-6
_POINT_IN_FRONT_OF_PLANE = 1
_POINT_ON_PLANE = 0
_POINT_BEHIND_PLANE = -1


def compute_volumetric_weight(length_cm: float, width_cm: float, height_cm: float, divisor: float = 5000.0) -> float:
    """Computes volumetric weight in kg from dimensions in cm.
    Formula: L * W * H (cm) / 5000.
    """
    return (length_cm * width_cm * height_cm) / divisor


class Box:
    """General 3D Oriented Bounding Box compatible with Objectron."""

    def __init__(self, vertices: Optional[np.ndarray] = None):
        if vertices is None:
            vertices = self.scaled_axis_aligned_vertices(np.array([1.0, 1.0, 1.0]))
        self._vertices = np.asarray(vertices, dtype=np.float64)
        self._rotation: Optional[np.ndarray] = None
        self._translation: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None
        self._transformation: Optional[np.ndarray] = None
        self._volume: Optional[float] = None

    @classmethod
    def from_transformation(cls, rotation: np.ndarray, translation: np.ndarray, scale: np.ndarray) -> "Box":
        """Constructs an oriented bounding box from 3x3 rotation, 3x1 translation, and 3-dim scale."""
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        scale = np.asarray(scale, dtype=np.float64).flatten()

        if rotation.size == 3:
            rotation = rotation_util.from_rotvec(rotation.tolist()).as_matrix()
        elif rotation.shape != (3, 3):
            raise ValueError(f"Unsupported rotation shape: {rotation.shape}")

        scaled_box = cls.scaled_axis_aligned_vertices(scale)
        vertices = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float64)
        for i in range(NUM_KEYPOINTS):
            vertices[i, :] = (rotation @ scaled_box[i, :].reshape(3, 1) + translation).flatten()
        return cls(vertices=vertices)

    @classmethod
    def scaled_axis_aligned_vertices(cls, scale: np.ndarray) -> np.ndarray:
        """Returns an axis-aligned set of 9 keypoint vertices for a given scale (sx, sy, sz)."""
        scale = np.asarray(scale, dtype=np.float64).flatten()
        w = scale[0] / 2.0
        h = scale[1] / 2.0
        d = scale[2] / 2.0
        aabb = np.array([
            [0.0, 0.0, 0.0],
            [-w, -h, -d],
            [-w, -h, +d],
            [-w, +h, -d],
            [-w, +h, +d],
            [+w, -h, -d],
            [+w, -h, +d],
            [+w, +h, -d],
            [+w, +h, +d],
        ], dtype=np.float64)
        return aabb

    @classmethod
    def fit(cls, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimates box orientation (3x3), translation (3,), and scale (3,) from 9 vertices."""
        vertices = np.asarray(vertices, dtype=np.float64)
        scale = np.zeros(3, dtype=np.float64)
        for axis in range(3):
            for edge_id in range(4):
                begin, end = EDGES[axis * 4 + edge_id]
                scale[axis] += np.linalg.norm(vertices[begin, :] - vertices[end, :])
            scale[axis] /= 4.0

        x = cls.scaled_axis_aligned_vertices(scale)
        system = np.concatenate((x, np.ones((NUM_KEYPOINTS, 1))), axis=1)
        solution, _, _, _ = np.linalg.lstsq(system, vertices, rcond=None)
        orientation = solution[:3, :3].T
        # Orthogonalize orientation matrix via SVD
        u, _, vt = np.linalg.svd(orientation)
        orientation = u @ vt
        if np.linalg.det(orientation) < 0:
            u[:, -1] *= -1
            orientation = u @ vt
        translation = solution[3, :3]
        return orientation, translation, scale

    def apply_transformation(self, transformation: np.ndarray) -> "Box":
        """Applies SE(3) 4x4 transformation matrix."""
        if transformation.shape != (4, 4):
            raise ValueError("Transformation must be a 4x4 matrix")
        new_rot = transformation[:3, :3] @ self.rotation
        new_trans = transformation[:3, 3] + (transformation[:3, :3] @ self.translation)
        return Box.from_transformation(new_rot, new_trans, self.scale)

    def inside(self, point: np.ndarray) -> bool:
        """Tests whether a given 3D point is inside the oriented box."""
        point = np.asarray(point, dtype=np.float64).reshape(3, 1)
        inv_trans = np.linalg.inv(self.transformation)
        point_local = (inv_trans[:3, :3] @ point).flatten() + inv_trans[:3, 3]
        scale = self.scale
        for i in range(3):
            if abs(point_local[i]) > (scale[i] / 2.0 + 1e-7):
                return False
        return True

    def sample(self) -> np.ndarray:
        """Samples a 3D point uniformly inside this box."""
        point = np.random.uniform(-0.5, 0.5, 3) * self.scale
        return (self.rotation @ point.reshape(3, 1)).flatten() + self.translation

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def rotation(self) -> np.ndarray:
        if self._rotation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._rotation

    @property
    def translation(self) -> np.ndarray:
        if self._translation is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._translation

    @property
    def scale(self) -> np.ndarray:
        if self._scale is None:
            self._rotation, self._translation, self._scale = self.fit(self._vertices)
        return self._scale

    @property
    def dimensions_cm(self) -> Tuple[float, float, float]:
        """Returns length, width, height in cm (sorted descending: L >= W >= H)."""
        dims = sorted((self.scale * 100.0).tolist(), reverse=True)
        return dims[0], dims[1], dims[2]

    @property
    def volumetric_weight_kg(self) -> float:
        """Volumetric weight in kg."""
        l, w, h = self.dimensions_cm
        return compute_volumetric_weight(l, w, h)

    @property
    def volume(self) -> float:
        """Volume in cubic meters."""
        if self._volume is None:
            i = self._vertices[2, :] - self._vertices[1, :]
            j = self._vertices[3, :] - self._vertices[1, :]
            k = self._vertices[5, :] - self._vertices[1, :]
            sys = np.array([i, j, k])
            self._volume = float(abs(np.linalg.det(sys)))
        return self._volume

    @property
    def transformation(self) -> np.ndarray:
        if self._transformation is None:
            t = np.identity(4, dtype=np.float64)
            t[:3, :3] = self.rotation
            t[:3, 3] = self.translation
            self._transformation = t
        return self._transformation

    def project_to_image(self, projection_matrix: np.ndarray, view_matrix: np.ndarray, width: int, height: int) -> np.ndarray:
        """Projects 3D vertices (9x3) onto 2D image coordinates using Objectron OpenGL view & projection matrices."""
        points = self._vertices
        p_3d = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1).T
        p_3d_cam = view_matrix @ p_3d
        p_2d_proj = projection_matrix @ p_3d_cam
        p_2d_ndc = (p_2d_proj[:-1, :] / (p_2d_proj[-1:, :] + 1e-9)).T

        x = p_2d_ndc[:, 1]
        y = p_2d_ndc[:, 0]
        pixels = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)
        pixels[:, 0] = ((1.0 + x) * 0.5) * width
        pixels[:, 1] = ((1.0 + y) * 0.5) * height
        return pixels

    def project_pinhole(self, intrinsics: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
        """Standard pinhole camera projection: P = K [R | t] * X."""
        k = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        ext = np.asarray(extrinsics, dtype=np.float64).reshape(4, 4)
        points_homo = np.concatenate((self._vertices, np.ones((NUM_KEYPOINTS, 1))), axis=1).T
        cam_points = ext @ points_homo
        z = cam_points[2, :]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        proj = k @ (cam_points[:3, :] / z)
        return proj[:2, :].T


class IoU:
    """Exact Sutherland-Hodgman 3D IoU for oriented bounding boxes."""

    def __init__(self, box1: Box, box2: Box):
        self._box1 = box1
        self._box2 = box2
        self._intersection_points: List[np.ndarray] = []

    def iou(self) -> float:
        """Computes exact 3D IoU using Sutherland-Hodgman clipping."""
        self._intersection_points = []
        self._compute_intersection_points(self._box1, self._box2)
        self._compute_intersection_points(self._box2, self._box1)

        if len(self._intersection_points) >= 4:
            try:
                hull = sp.ConvexHull(self._intersection_points)
                intersection_volume = hull.volume
                box1_vol = self._box1.volume
                box2_vol = self._box2.volume
                union_volume = box1_vol + box2_vol - intersection_volume
                if union_volume <= 1e-9:
                    return 0.0
                val = float(intersection_volume / union_volume)
                return max(0.0, min(1.0, val))
            except Exception:
                return self.iou_sampling(2000)
        else:
            return 0.0

    def iou_sampling(self, num_samples: int = 5000) -> float:
        """Monte Carlo sampling approximation for 3D IoU."""
        p1 = [self._box1.sample() for _ in range(num_samples)]
        p2 = [self._box2.sample() for _ in range(num_samples)]
        b1_vol = self._box1.volume
        b2_vol = self._box2.volume

        b1_inside_b2 = sum(1 for p in p1 if self._box2.inside(p))
        b2_inside_b1 = sum(1 for p in p2 if self._box1.inside(p))

        intersect_vol = (b1_vol * b1_inside_b2 + b2_vol * b2_inside_b1) / (2.0 * num_samples)
        union_vol = b1_vol + b2_vol - intersect_vol
        if union_vol <= 1e-9:
            return 0.0
        return float(max(0.0, min(1.0, intersect_vol / union_vol)))

    def _compute_intersection_points(self, box_src: Box, box_template: Box) -> None:
        inv_trans = np.linalg.inv(box_src.transformation)
        box_src_aligned = box_src.apply_transformation(inv_trans)
        template_in_src = box_template.apply_transformation(inv_trans)

        for face in range(len(FACES)):
            indices = FACES[face, :]
            poly = [template_in_src.vertices[indices[i], :] for i in range(4)]
            clip = self.intersect_box_poly(box_src_aligned, poly)
            for point in clip:
                point_w = (box_src.rotation @ point.reshape(3, 1)).flatten() + box_src.translation
                self._intersection_points.append(point_w)

        for point_id in range(NUM_KEYPOINTS):
            v = template_in_src.vertices[point_id, :]
            if box_src_aligned.inside(v):
                point_w = (box_src.rotation @ v.reshape(3, 1)).flatten() + box_src.translation
                self._intersection_points.append(point_w)

    def intersect_box_poly(self, box: Box, poly: List[np.ndarray]) -> List[np.ndarray]:
        for face in range(len(FACES)):
            indices = FACES[face, :]
            plane_v = box.vertices[indices[0], :]
            # compute plane normal
            v1 = box.vertices[indices[1], :] - plane_v
            v2 = box.vertices[indices[2], :] - plane_v
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            normal = normal / norm
            poly = self.clip_poly_plane(poly, plane_v, normal)
            if not poly:
                break
        return poly

    def clip_poly_plane(self, poly: List[np.ndarray], plane_pt: np.ndarray, plane_normal: np.ndarray) -> List[np.ndarray]:
        clipped: List[np.ndarray] = []
        if not poly:
            return clipped

        def classify_point(p: np.ndarray) -> int:
            dist = np.dot(p - plane_pt, plane_normal)
            if dist > _PLANE_THICKNESS_EPSILON:
                return _POINT_IN_FRONT_OF_PLANE
            elif dist < -_PLANE_THICKNESS_EPSILON:
                return _POINT_BEHIND_PLANE
            return _POINT_ON_PLANE

        p1 = poly[-1]
        c1 = classify_point(p1)
        for p2 in poly:
            c2 = classify_point(p2)
            if c2 == _POINT_IN_FRONT_OF_PLANE:
                if c1 == _POINT_BEHIND_PLANE:
                    t = np.dot(plane_pt - p1, plane_normal) / (np.dot(p2 - p1, plane_normal) + 1e-12)
                    clipped.append(p1 + t * (p2 - p1))
                clipped.append(p2)
            elif c2 == _POINT_BEHIND_PLANE:
                if c1 == _POINT_IN_FRONT_OF_PLANE:
                    t = np.dot(plane_pt - p1, plane_normal) / (np.dot(p2 - p1, plane_normal) + 1e-12)
                    clipped.append(p1 + t * (p2 - p1))
            else:
                clipped.append(p2)
            p1 = p2
            c1 = c2
        return clipped
