"""Unified Box Dimensioning Pipeline Orchestrator and Output Visualizer (F6).
Executes Method A or Method B, performs quality gating, renders reprojected 3D wireframe,
and outputs standardized JSON specification for volumetric-weight courier billing.
"""

from enum import Enum
from typing import Dict, Optional, Tuple
import os
import json
import cv2
import numpy as np

from box_dimensioning.dataset.box_geometry import Box, IoU
from box_dimensioning.dataset.objectron_parser import ObjectronSequence
from box_dimensioning.calibration.intrinsics_checker import draw_projected_box_wireframe
from box_dimensioning.pipeline.method_a_multiview import MethodAMultiView, MeasurementResult
from box_dimensioning.pipeline.method_b_singleview import MethodBSingleView


class DimensioningMethod(str, Enum):
    """Canonical dimensioning pipeline identifiers."""
    METHOD_A = "METHOD_A_MULTIVIEW"
    METHOD_B = "METHOD_B_SINGLEVIEW"


_METHOD_ALIASES: Dict[str, DimensioningMethod] = {
    "A": DimensioningMethod.METHOD_A,
    "METHOD_A": DimensioningMethod.METHOD_A,
    "METHOD_A_MULTIVIEW": DimensioningMethod.METHOD_A,
    "B": DimensioningMethod.METHOD_B,
    "METHOD_B": DimensioningMethod.METHOD_B,
    "METHOD_B_SINGLEVIEW": DimensioningMethod.METHOD_B,
}


def normalize_method(method: str) -> DimensioningMethod:
    """Normalizes user-supplied method string to a canonical DimensioningMethod."""
    if not isinstance(method, str):
        raise ValueError(f"Method must be a string, got {type(method).__name__}")
    key = method.strip().upper()
    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unknown dimensioning method '{method}'. Expected one of: A, B, METHOD_A_MULTIVIEW, METHOD_B_SINGLEVIEW."
        )
    return _METHOD_ALIASES[key]


def render_wireframe_with_overlay(
    image: np.ndarray,
    projected_wireframe_2d: np.ndarray,
    dimensions_cm: Dict[str, float],
    volumetric_weight_kg: float,
    status: str,
    method: str,
    gt_dimensions_cm: Optional[Dict[str, float]] = None,
    iou_3d: Optional[float] = None
) -> np.ndarray:
    """Renders 3D wireframe and diagnostic telemetry card onto the video frame."""
    annotated = draw_projected_box_wireframe(
        image,
        projected_wireframe_2d,
        color_edges=(0, 255, 0) if status == "ACCEPTED" else (0, 0, 255),
        color_corners=(0, 255, 255),
        thickness=3
    )

    # Highlight all eight projected 3D cuboid corners (keypoints 1 through 8)
    pts_int = np.asarray(projected_wireframe_2d, dtype=np.int32)
    for c_i in range(1, min(9, len(pts_int))):
        pt = tuple(pts_int[c_i])
        cv2.circle(annotated, pt, 7, (0, 0, 0), -1)
        cv2.circle(annotated, pt, 5, (0, 255, 255), -1)
        cv2.putText(annotated, f"C{c_i}", (pt[0] + 6, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    h, w = annotated.shape[:2]
    card_w, card_h = min(540, w - 20), 220
    overlay = annotated.copy()

    # Draw semi-transparent HUD card in top-left corner
    cv2.rectangle(overlay, (15, 15), (15 + card_w, 15 + card_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
    cv2.rectangle(annotated, (15, 15), (15 + card_w, 15 + card_h), (0, 200, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    l = dimensions_cm.get("length", 0.0)
    width = dimensions_cm.get("width", 0.0)
    height = dimensions_cm.get("height", 0.0)

    # Status indicator
    status_color = (0, 255, 0) if status == "ACCEPTED" else (0, 0, 255)
    cv2.putText(annotated, f"STATUS: {status}", (30, 48), font, 0.75, status_color, 2)
    cv2.putText(annotated, f"Method: {method}", (30, 78), font, 0.55, (220, 220, 220), 1)

    # Measured dimensions and volumetric weight
    dim_str = f"Dims: L={l:.1f} x W={width:.1f} x H={height:.1f} cm"
    cv2.putText(annotated, dim_str, (30, 110), font, 0.65, (255, 255, 255), 2)
    vw_str = f"Volumetric Weight: {volumetric_weight_kg:.3f} kg (L*W*H/5000)"
    cv2.putText(annotated, vw_str, (30, 140), font, 0.65, (0, 255, 255), 2)

    # Ground truth comparison if available
    if gt_dimensions_cm:
        gt_l = gt_dimensions_cm.get("length", 0.0)
        gt_w = gt_dimensions_cm.get("width", 0.0)
        gt_h = gt_dimensions_cm.get("height", 0.0)
        gt_str = f"GT:   L={gt_l:.1f} x W={gt_w:.1f} x H={gt_h:.1f} cm"
        cv2.putText(annotated, gt_str, (30, 175), font, 0.55, (180, 180, 180), 1)

    if iou_3d is not None:
        iou_str = f"3D IoU (Sutherland-Hodgman): {iou_3d:.2f}"
        cv2.putText(annotated, iou_str, (30, 205), font, 0.55, (0, 255, 0) if iou_3d >= 0.50 else (0, 165, 255), 2)

    return annotated


class BoxDimensioner:
    """High-level box dimensioning system."""

    def __init__(self):
        self.method_a = MethodAMultiView()
        self.method_b = MethodBSingleView()

    def dimension_sequence(
        self,
        sequence: ObjectronSequence,
        images: Optional[Dict[int, np.ndarray]] = None,
        method: str = "METHOD_A_MULTIVIEW",
        output_image_path: Optional[str] = None
    ) -> Dict:
        """Runs dimensioning measurement on an Objectron video sequence.
        Produces complete output JSON payload and wireframe image (F6).
        """
        canonical_method = normalize_method(method)
        if canonical_method == DimensioningMethod.METHOD_B:
            res: MeasurementResult = self.method_b.measure(sequence, images)
        else:
            res: MeasurementResult = self.method_a.measure(sequence, images)

        # Build output dictionary
        output = {
            "sequence_id": sequence.sequence_id,
            "status": res.status,
            "method": res.method,
            "dimensions_cm": res.dimensions_cm,
            "volumetric_weight_kg": res.volumetric_weight_kg,
            "quality_flags": {
                "is_truncated": res.quality_report.is_truncated,
                "blur_score": round(res.quality_report.blur_score, 2),
                "is_blurry": res.quality_report.is_blurry,
                "baseline_meters": res.quality_report.baseline_meters,
                "fit_residual": res.fit_residual,
                "reason_code": res.quality_report.reason_code,
            },
            "selected_frames": res.selected_frame_indices,
            "latency_seconds": res.latency_seconds,
            "reprojected_frame_index": res.reprojected_frame_index,
            "reprojected_wireframe_2d": res.reprojected_wireframe_2d,
            "measured_box_3d": res.measured_box_3d,
        }

        # Compare with ground-truth for evaluation (ground truth NEVER used at inference)
        gt_dims_dict = None
        iou_val = None
        if sequence.ground_truth is not None and res.measured_box_3d is not None:
            gt_l, gt_w, gt_h = sequence.ground_truth.dimensions_cm
            gt_vw = sequence.ground_truth.volumetric_weight_kg
            gt_dims_dict = {"length": round(gt_l, 2), "width": round(gt_w, 2), "height": round(gt_h, 2)}

            meas_l = res.dimensions_cm["length"]
            meas_w = res.dimensions_cm["width"]
            meas_h = res.dimensions_cm["height"]

            err_l = abs(meas_l - gt_l)
            err_w = abs(meas_w - gt_w)
            err_h = abs(meas_h - gt_h)
            mae = (err_l + err_w + err_h) / 3.0

            vw_err_pct = ((res.volumetric_weight_kg - gt_vw) / max(1e-6, gt_vw)) * 100.0

            # 3D IoU calculation
            iou_calc = IoU(res.measured_box_3d, sequence.ground_truth.box_3d)
            iou_val = round(iou_calc.iou(), 4)

            output["evaluation"] = {
                "ground_truth_dimensions_cm": gt_dims_dict,
                "ground_truth_volumetric_weight_kg": round(gt_vw, 4),
                "dimension_errors_cm": {
                    "error_length": round(err_l, 2),
                    "error_width": round(err_w, 2),
                    "error_height": round(err_h, 2),
                    "mae_cm": round(mae, 2),
                },
                "volumetric_weight_error_pct": round(vw_err_pct, 2),
                "volumetric_weight_within_15_pct": abs(vw_err_pct) <= 15.0,
                "iou_3d": iou_val,
                "iou_3d_ge_0_50": (iou_val >= 0.50),
            }

        # Render output image if image provided or path requested
        target_frame_idx = res.reprojected_frame_index
        target_img = images.get(target_frame_idx) if images else None

        # Look up true original video frame from the sequence if not explicitly passed
        if target_img is None:
            target_img = sequence.get_image(target_frame_idx)

        # If original frame is available, render true 3D wireframe overlay on the real frame
        if target_img is not None and res.reprojected_wireframe_2d is not None and output_image_path is not None:
            annotated_img = render_wireframe_with_overlay(
                image=target_img,
                projected_wireframe_2d=res.reprojected_wireframe_2d,
                dimensions_cm=res.dimensions_cm,
                volumetric_weight_kg=res.volumetric_weight_kg,
                status=res.status,
                method=res.method,
                gt_dimensions_cm=gt_dims_dict,
                iou_3d=iou_val
            )
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, annotated_img)
            output["reprojection_image_path"] = output_image_path
        else:
            # Honest reporting: do NOT create fake black image
            output["reprojection_image_path"] = None

        return output
