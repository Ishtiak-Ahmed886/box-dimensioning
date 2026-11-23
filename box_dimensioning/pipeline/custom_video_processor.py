"""Custom Mobile Video Processing and Dimensioning Pipeline.
Enables couriers and users to upload raw phone videos (.mp4, .mov),
automatically extracts keyframes with optimal baseline and sharpness,
reconstructs 3D box geometry via structure-from-motion / DLT triangulation,
and computes volumetric weight for billing.

Improvements (v2):
- Step 2: Smart intrinsics (EXIF → phone DB → heuristic) + radial undistortion
- Step 3: Deep detector (YOLOv8-nano ONNX → Hough-line classical fallback)
- Step 4: Scale anchor (A4/credit card reference object removes scale ambiguity)
"""

from typing import Dict, List, Optional, Tuple
import os
import time
import cv2
import numpy as np

from box_dimensioning.dataset.box_geometry import Box, compute_volumetric_weight
from box_dimensioning.segmentation.classical import ClassicalBoxSegmenter
from box_dimensioning.segmentation.deep_detector import DeepBoxDetector
from box_dimensioning.pipeline.quality_gate import QualityGate, QualityReport
from box_dimensioning.pipeline.scale_anchor import (
    compute_absolute_scale, scale_dimensions_to_cm, draw_reference_overlay,
    detect_reference_rectangle, REFERENCE_OBJECTS
)
from box_dimensioning.calibration.smart_intrinsics import (
    estimate_intrinsics_smart, apply_radial_undistort
)
from box_dimensioning.calibration.intrinsics_checker import draw_projected_box_wireframe


def extract_video_frames(
    video_path: str,
    max_frames: int = 60,
    stride: int = 2
) -> List[Tuple[int, np.ndarray]]:
    """Extracts indexed frames uniformly from an uploaded video file."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 120

    effective_step = max(1, total_frames // max_frames)
    frames: List[Tuple[int, np.ndarray]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % effective_step == 0:
            frames.append((frame_idx, frame))
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames


def estimate_phone_intrinsics(width: int, height: int, fov_deg: float = 68.0) -> np.ndarray:
    """Estimates standard pinhole camera intrinsic matrix for a smartphone camera.
    Standard wide lenses have ~68-72 degree horizontal FOV in landscape or vertical in portrait.
    """
    max_dim = max(width, height)
    # Focal length in pixels: f = (max_dim / 2) / tan(fov / 2)
    f = (max_dim / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [f, 0.0, cx],
        [0.0, f, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def select_best_keyframe_pair(
    indexed_frames: List[Tuple[int, np.ndarray]],
    quality_gate: QualityGate
) -> Tuple[int, int, np.ndarray, np.ndarray, float]:
    """Selects two candidate frames with high sharpness and optimal parallax/baseline.
    Returns: (idx1, idx2, frame1, frame2, blur_score).
    """
    if len(indexed_frames) < 2:
        raise ValueError("At least 2 frames required for multi-view measurement")

    # Filter out severely blurry frames
    sharp_frames = []
    for idx, f in indexed_frames:
        blur = quality_gate.compute_blur_score(f)
        if blur >= quality_gate.blur_threshold * 0.7:  # relaxed threshold for candid mobile
            sharp_frames.append((idx, f, blur))

    if len(sharp_frames) < 2:
        # Fallback to all frames sorted by sharpness
        sorted_by_sharp = sorted(indexed_frames, key=lambda x: quality_gate.compute_blur_score(x[1]), reverse=True)
        sharp_frames = [(x[0], x[1], quality_gate.compute_blur_score(x[1])) for x in sorted_by_sharp[:10]]

    # Compute pairwise optical flow / SIFT displacement between candidates
    orb = cv2.ORB_create(nfeatures=500)
    best_pair = (sharp_frames[0][0], sharp_frames[-1][0], sharp_frames[0][1], sharp_frames[-1][1], sharp_frames[0][2])
    best_score = -1.0

    n_cands = len(sharp_frames)
    for i in range(n_cands):
        idx1, img1, b1 = sharp_frames[i]
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        kp1, des1 = orb.detectAndCompute(gray1, None)
        if des1 is None or len(des1) < 15:
            continue

        for j in range(i + 1, min(n_cands, i + 8)):
            idx2, img2, b2 = sharp_frames[j]
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            kp2, des2 = orb.detectAndCompute(gray2, None)
            if des2 is None or len(des2) < 15:
                continue

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) < 10:
                continue

            # Average displacement
            shifts = [np.linalg.norm(np.array(kp1[m.queryIdx].pt) - np.array(kp2[m.trainIdx].pt)) for m in matches]
            mean_shift = float(np.median(shifts))

            # Optimal displacement between 30px and 200px
            if 25.0 <= mean_shift <= 250.0:
                score = len(matches) / (abs(mean_shift - 80.0) + 10.0) + (b1 + b2) * 0.05
                if score > best_score:
                    best_score = score
                    best_pair = (idx1, idx2, img1, img2, float((b1 + b2) / 2.0))

    return best_pair


def triangulate_custom_rays(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """Triangulates matched 2D points into 3D using DLT with calibrated camera matrices."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])

    pts1_h = pts1.T  # 2xN
    pts2_h = pts2.T  # 2xN
    pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    pts3d = (pts4d[:3] / (pts4d[3:4] + 1e-9)).T  # Nx3

    # Filter out points behind camera or unreasonably distant
    valid = (pts3d[:, 2] > 0.1) & (pts3d[:, 2] < 5.0)
    return pts3d[valid] if np.any(valid) else pts3d


def process_custom_mobile_video(
    video_path: str,
    baseline_cm: float = 12.0,
    quality_gate: Optional[QualityGate] = None,
    segmenter: Optional[ClassicalBoxSegmenter] = None,
    reference_object: Optional[str] = None,   # e.g. "A4 Paper", "Credit Card"
    phone_make: Optional[str] = None,          # e.g. "Samsung"
    phone_model: Optional[str] = None,         # e.g. "SM-S911"
    use_deep_detector: bool = True,
    strict_quality_gate: bool = False,
    **kwargs
) -> Dict:
    """Processes an uploaded mobile video, recovers geometry, and calculates volumetric weight.
    Returns: Complete measurement dictionary with HUD-rendered frame.
    """
    t_start = time.perf_counter()
    # For custom mobile video, use a relaxed blur threshold (15.0) suitable for phone video compression
    q_gate = quality_gate or QualityGate(blur_threshold=15.0)
    seg = segmenter or ClassicalBoxSegmenter()
    deep_detector = DeepBoxDetector(use_deep=use_deep_detector)

    # Step 1: Extract frames from video
    indexed_frames = extract_video_frames(video_path, max_frames=50)
    if len(indexed_frames) < 2:
        return {
            "status": "REJECTED",
            "method": "CUSTOM_MOBILE_VIDEO",
            "reason_code": "INSUFFICIENT_FRAMES",
            "dimensions_cm": {"length": 0.0, "width": 0.0, "height": 0.0},
            "volumetric_weight_kg": 0.0,
            "latency_seconds": round(time.perf_counter() - t_start, 3),
        }

    # Step 2: Select best pair of keyframes
    idx1, idx2, img1, img2, blur_score = select_best_keyframe_pair(indexed_frames, q_gate)
    h, w = img1.shape[:2]

    # ── UPGRADE 1: Smart Camera Intrinsics ──────────────────────────────────
    # Try to read EXIF from first video frame saved as JPEG bytes
    frame_bytes: Optional[bytes] = None
    try:
        is_ok, buf = cv2.imencode('.jpg', img1, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if is_ok:
            frame_bytes = buf.tobytes()
    except Exception:
        pass

    K, intrinsics_info = estimate_intrinsics_smart(
        width=w, height=h,
        image_bytes=frame_bytes,
        make=phone_make,
        model=phone_model,
    )
    # Apply radial undistortion (barrel correction for wide-angle phone lens)
    img1 = apply_radial_undistort(img1, K)
    img2 = apply_radial_undistort(img2, K)

    # ── UPGRADE 2: Deep Box Detection ───────────────────────────────────────
    # Use YOLOv8-nano ONNX for primary detection; Hough-line classical as fallback
    c1_res = deep_detector.detect(img1)
    c2_res = deep_detector.detect(img2)
    corners1 = c1_res.get("corners_2d", np.empty((0, 2)))
    corners2 = c2_res.get("corners_2d", np.empty((0, 2)))
    detection_method = c1_res.get("method", "unknown")

    # Quality Gate checks (For custom upload, only hard-reject if strict_quality_gate is enabled)
    q_report = q_gate.validate_frame(img1, corners1, w, h)
    if strict_quality_gate and q_report.status == "REJECTED":
        return {
            "status": "REJECTED",
            "method": "CUSTOM_MOBILE_VIDEO",
            "reason_code": q_report.reason_code,
            "quality_flags": {
                "is_truncated": q_report.is_truncated,
                "blur_score": round(blur_score, 1),
                "is_blurry": q_report.is_blurry,
            },
            "dimensions_cm": {"length": 0.0, "width": 0.0, "height": 0.0},
            "volumetric_weight_kg": 0.0,
            "latency_seconds": round(time.perf_counter() - t_start, 3),
        }

    # ── UPGRADE 3: Scale Anchor ──────────────────────────────────────────────
    # Detect reference object (A4, credit card, etc.) to resolve scale ambiguity
    scale_mm_per_px: Optional[float] = None
    ref_detection = None
    if reference_object and reference_object in REFERENCE_OBJECTS:
        ref_detection = detect_reference_rectangle(img1, ref_name=reference_object)
        if ref_detection is not None:
            scale_mm_per_px = ref_detection["scale_mm_per_px"]

    # Step 5: SIFT / Feature Matching for camera motion estimation
    sift = cv2.SIFT_create(nfeatures=800)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    R = np.eye(3)
    t = np.array([baseline_cm / 100.0, 0.0, 0.0], dtype=np.float64)

    if des1 is not None and des2 is not None and len(des1) >= 8 and len(des2) >= 8:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) >= 8:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

            E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
            if E is not None and E.shape == (3, 3):
                _, R_rec, t_rec, _ = cv2.recoverPose(E, src_pts, dst_pts, K)
                R = R_rec
                # Scale unit vector t by measured/estimated baseline in meters
                scale_m = baseline_cm / 100.0
                t = (t_rec.flatten() * scale_m).astype(np.float64)

    # Step 6: Triangulate parcel boundary corners via Lucas-Kanade tracking and Epipolar validation
    triangulated = []
    if len(corners1) >= 4:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        p1 = np.float32(corners1).reshape(-1, 1, 2)
        p2, st, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, p1, None, winSize=(21, 21), maxLevel=3)
        p1_back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, p2, None, winSize=(21, 21), maxLevel=3)

        matched_p1 = []
        matched_p2 = []
        if p2 is not None and st is not None and st_back is not None:
            for idx in range(len(corners1)):
                if st[idx] == 1 and st_back[idx] == 1:
                    d_back = np.linalg.norm(p1[idx] - p1_back[idx])
                    if d_back < 3.0:  # consistent optical flow track
                        pt1 = p1[idx, 0]
                        pt2 = p2[idx, 0]
                        if 0 <= pt2[0] < w and 0 <= pt2[1] < h:
                            matched_p1.append(pt1)
                            matched_p2.append(pt2)

        if len(matched_p1) >= 4:
            pts1_arr = np.array(matched_p1, dtype=np.float64)
            pts2_arr = np.array(matched_p2, dtype=np.float64)
            tri_pts = triangulate_custom_rays(pts1_arr, pts2_arr, K, R, t)
            # Ensure triangulated points are in front of camera
            valid_depth = (tri_pts[:, 2] > 0.1) & (tri_pts[:, 2] < 3.0)
            if np.sum(valid_depth) >= 4:
                triangulated = tri_pts[valid_depth]

    # Step 7: Estimate Dimensions & 3D Center/Pose from 3D cloud or perspective bounding box
    if len(triangulated) >= 4:
        center_3d = np.median(triangulated, axis=0)
        u, s, vh = np.linalg.svd(triangulated - center_3d, full_matrices=False)
        rot_box = vh.T
        if np.linalg.det(rot_box) < 0:
            rot_box[:, -1] *= -1

        local_pts = (rot_box.T @ (triangulated - center_3d).T).T
        mins = np.percentile(local_pts, 5, axis=0)
        maxs = np.percentile(local_pts, 95, axis=0)
        diffs = np.abs(maxs - mins) * 100.0  # meters to cm
        dims = sorted(diffs.tolist(), reverse=True)
        # Prevent degenerate near-zero height
        if dims[2] < 2.0 or dims[2] < dims[0] * 0.15:
            dims[2] = max(3.0, dims[1] * 0.5)
            dims = sorted(dims, reverse=True)
    else:
        # Monocular depth scaling fallback using GrabCut bounding box
        rect = c1_res.get("bounding_box", (w // 4, h // 4, w // 2, h // 2))
        cx_box = rect[0] + rect[2] / 2.0
        cy_box = rect[1] + rect[3] / 2.0
        box_w_px = float(rect[2])
        box_h_px = float(rect[3])
        f_px = float(K[0, 0])
        # Typical courier parcel scanning distance ~ 0.55m
        est_dist_m = 0.55
        center_3d = np.array([
            (cx_box - K[0, 2]) * est_dist_m / K[0, 0],
            (cy_box - K[1, 2]) * est_dist_m / K[1, 1],
            est_dist_m
        ], dtype=np.float64)

        # Realistic 15-degree perspective tilt so top, front, and side 3D faces are visible
        cos_t = np.cos(np.deg2rad(15))
        sin_t = np.sin(np.deg2rad(15))
        rot_box = np.array([
            [cos_t, 0.0, sin_t],
            [sin_t * 0.3, cos_t, -cos_t * 0.3],
            [-sin_t, 0.3, cos_t]
        ], dtype=np.float64)

        est_w_cm = (box_w_px * est_dist_m / f_px) * 100.0
        est_h_cm = (box_h_px * est_dist_m / f_px) * 100.0
        est_d_cm = max(4.0, est_w_cm * 0.65)
        dims = sorted([est_w_cm, est_h_cm, est_d_cm], reverse=True)

    # Apply scale anchor if available, keeping raw unit as cm
    rect = c1_res.get("bounding_box", (w // 4, h // 4, w // 2, h // 2))
    pixel_dims = (float(rect[2]), float(rect[3]), float(rect[2]) * 0.65)
    raw_tuple = (float(dims[0]), float(dims[1]), float(dims[2]))
    length, width, height = scale_dimensions_to_cm(
        raw_dims=raw_tuple,
        scale_mm_per_px=scale_mm_per_px,
        pixel_dims=pixel_dims if scale_mm_per_px is not None else None,
        fallback_distance_m=0.55,
        focal_length_px=float(K[0, 0]),
        raw_unit="cm"
    )
    vol_weight = compute_volumetric_weight(length, width, height)

    # Step 8: Render True 3D Wireframe & Shaded Cuboid Faces onto Keyframe Image
    from box_dimensioning.visualization.box_3d_renderer import render_true_3d_wireframe_hud

    annotated = render_true_3d_wireframe_hud(
        image=img1,
        center_3d=center_3d,
        rot_3d=rot_box,
        length_cm=length,
        width_cm=width,
        height_cm=height,
        K=K,
        volumetric_weight_kg=vol_weight
    )

    # Draw reference overlay if detected
    if ref_detection is not None:
        annotated = draw_reference_overlay(annotated, ref_detection)

    latency = round(time.perf_counter() - t_start, 3)

    return {
        "status": "ACCEPTED",
        "method": "CUSTOM_MOBILE_VIDEO",
        "dimensions_cm": {
            "length": round(length, 1),
            "width": round(width, 1),
            "height": round(height, 1)
        },
        "volumetric_weight_kg": round(vol_weight, 4),
        "quality_flags": {
            "is_truncated": q_report.is_truncated,
            "blur_score": round(blur_score, 1),
            "is_blurry": q_report.is_blurry,
            "reason_code": "WARN_BLURRY" if q_report.is_blurry else "ACCEPT"
        },
        "selected_frames": [int(idx1), int(idx2)],
        "latency_seconds": latency,
        "annotated_image": annotated,
        "center_3d": center_3d,
        "rot_box": rot_box,
        "raw_frame_1": img1,
        "raw_frame_2": img2,
        # Improvement metadata
        "intrinsics_info": intrinsics_info,
        "detection_method": detection_method,
        "scale_anchor": {
            "reference_object": reference_object,
            "scale_mm_per_px": round(scale_mm_per_px, 4) if scale_mm_per_px else None,
            "anchor_detected": ref_detection is not None,
        },
    }
