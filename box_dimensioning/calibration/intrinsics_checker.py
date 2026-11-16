"""Sanity-check Objectron intrinsics by reprojecting an annotated 3D box onto video frames (F1)."""

from typing import Dict, Tuple, Optional
import os
import cv2
import numpy as np

from box_dimensioning.dataset.box_geometry import Box, EDGES, NUM_KEYPOINTS
from box_dimensioning.dataset.objectron_parser import ObjectronSequence, FrameGeometry, GroundTruthBox


def draw_projected_box_wireframe(
    image: np.ndarray,
    projected_points_2d: np.ndarray,
    color_edges: Tuple[int, int, int] = (0, 255, 0),
    color_corners: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2
) -> np.ndarray:
    """Draws 3D wireframe bounding box on an image given 9 2D keypoint coordinates."""
    img = image.copy()
    pts = np.asarray(projected_points_2d, dtype=np.int32)

    # Draw edges
    for begin, end in EDGES:
        pt1 = tuple(pts[begin])
        pt2 = tuple(pts[end])
        cv2.line(img, pt1, pt2, color_edges, thickness)

    # Draw corners
    for i in range(1, NUM_KEYPOINTS):
        cv2.circle(img, tuple(pts[i]), thickness + 2, color_corners, -1)

    # Draw center point (0)
    cv2.circle(img, tuple(pts[0]), thickness + 3, (255, 0, 0), -1)

    return img


def check_objectron_intrinsics(
    sequence: ObjectronSequence,
    frame_index: int = 0,
    annotated_kps_2d: Optional[np.ndarray] = None,
    frame_image: Optional[np.ndarray] = None,
    output_vis_path: Optional[str] = None
) -> Dict:
    """Sanity checks Objectron camera projection geometry against annotated 2D keypoints."""
    if sequence.ground_truth is None:
        raise ValueError("Sequence does not contain ground truth box")
    if frame_index >= len(sequence.frames):
        raise IndexError(f"Frame index {frame_index} out of range (total {len(sequence.frames)})")

    frame: FrameGeometry = sequence.get_frame(frame_index)
    gt: GroundTruthBox = sequence.ground_truth

    # Project 3D box to 2D image coordinates using Objectron OpenGL view & projection matrices
    proj_2d = gt.box_3d.project_to_image(
        projection_matrix=frame.projection_matrix,
        view_matrix=frame.view_matrix,
        width=frame.image_width,
        height=frame.image_height
    )

    result = {
        "sequence_id": sequence.sequence_id,
        "frame_index": frame_index,
        "image_resolution": [frame.image_width, frame.image_height],
        "projected_keypoints_2d": proj_2d.tolist(),
        "sanity_check_passed": True,
    }

    # If annotated 2D keypoints are provided, compare Euclidean pixel error
    if annotated_kps_2d is not None and len(annotated_kps_2d) >= NUM_KEYPOINTS:
        # Scale normalized [0, 1] keypoints to pixel coords if needed
        annot_pts = np.array(annotated_kps_2d[:NUM_KEYPOINTS, :2], dtype=np.float64)
        if np.max(annot_pts) < 10.0:
            annot_pts[:, 0] *= frame.image_width
            annot_pts[:, 1] *= frame.image_height

        # Keypoint 0 in Objectron is centroid, 1-8 are bounding box vertices
        # Compare vertices (1 to 8)
        diffs = np.linalg.norm(proj_2d[1:] - annot_pts[1:], axis=1)
        mean_err = float(np.mean(diffs))
        max_err = float(np.max(diffs))

        result["mean_reprojection_error_px"] = mean_err
        result["max_reprojection_error_px"] = max_err
        # Sanity check passes if mean error is within reasonable reprojection tolerance (< 15 px for full resolution)
        result["sanity_check_passed"] = bool(mean_err < 15.0)

    # Visualize and save wireframe if image is provided
    if frame_image is not None and output_vis_path is not None:
        vis_img = draw_projected_box_wireframe(frame_image, proj_2d)
        os.makedirs(os.path.dirname(output_vis_path), exist_ok=True)
        cv2.imwrite(output_vis_path, vis_img)
        result["visualization_path"] = output_vis_path

    return result
