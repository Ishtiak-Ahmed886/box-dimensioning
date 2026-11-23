"""Evaluation metrics for box dimensioning and volumetric-weight billing (A1-A7).
Computes:
- A2: MAE per dimension (cm)
- A3: Volumetric weight accuracy (% within +-15%)
- A4: 3D IoU (Sutherland-Hodgman)
- A5: Comparative evaluation (Method A vs Method B)
- A6: Quality gate rejection statistics (TPR >= 90%, FPR <= 10%)
- A7: Execution latency (<= 2.0 s on CPU)
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from box_dimensioning.dataset.box_geometry import Box, IoU


def compute_dimension_errors(
    measured_dims_cm: Dict[str, float],
    gt_dims_cm: Dict[str, float]
) -> Dict:
    """Computes per-dimension errors and Mean Absolute Error (MAE) in cm."""
    err_l = abs(measured_dims_cm["length"] - gt_dims_cm["length"])
    err_w = abs(measured_dims_cm["width"] - gt_dims_cm["width"])
    err_h = abs(measured_dims_cm["height"] - gt_dims_cm["height"])
    mae = (err_l + err_w + err_h) / 3.0

    return {
        "error_length_cm": round(err_l, 3),
        "error_width_cm": round(err_w, 3),
        "error_height_cm": round(err_h, 3),
        "mae_cm": round(mae, 3),
        "meets_A2_threshold": mae <= 2.0,  # A2: MAE <= 2.0 cm
    }


def compute_volumetric_weight_metrics(
    measured_vw_kg: float,
    gt_vw_kg: float,
    tolerance_pct: float = 15.0
) -> Dict:
    """Computes volumetric weight error percentage and verifies acceptance criterion A3."""
    diff = measured_vw_kg - gt_vw_kg
    err_pct = (abs(diff) / max(1e-6, gt_vw_kg)) * 100.0

    return {
        "measured_vw_kg": round(measured_vw_kg, 4),
        "gt_vw_kg": round(gt_vw_kg, 4),
        "absolute_error_kg": round(abs(diff), 4),
        "error_pct": round(err_pct, 2),
        "within_15_pct": err_pct <= tolerance_pct,
    }


def compute_3d_iou_metric(box_measured: Box, box_gt: Box) -> Dict:
    """Computes exact 3D IoU using Sutherland-Hodgman algorithm and evaluates criterion A4."""
    iou_calc = IoU(box_measured, box_gt)
    val = iou_calc.iou()
    return {
        "iou_3d": round(val, 4),
        "meets_A4_threshold": val >= 0.50,  # A4: Mean 3D IoU >= 0.50
    }


def compute_quality_gate_confusion_matrix(
    gt_is_truncated: List[bool],
    gate_rejected_for_truncation: List[bool]
) -> Dict:
    """Evaluates Quality Gate rejection performance for Acceptance Criterion A6.
    Requirement A6: Rejects >= 90% of truncated frames (TPR >= 0.90)
    with <= 10% false rejections of fully visible frames (FPR <= 0.10).
    """
    tp, fp, tn, fn = 0, 0, 0, 0
    for gt_trunc, rejected in zip(gt_is_truncated, gate_rejected_for_truncation):
        if gt_trunc and rejected:
            tp += 1
        elif not gt_trunc and rejected:
            fp += 1
        elif not gt_trunc and not rejected:
            tn += 1
        else:
            fn += 1

    total_truncated = tp + fn
    total_visible = fp + tn

    tpr = (tp / max(1, total_truncated))
    fpr = (fp / max(1, total_visible))
    precision = (tp / max(1, tp + fp))
    accuracy = ((tp + tn) / max(1, tp + fp + tn + fn))

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_truncated_frames": total_truncated,
        "total_visible_frames": total_visible,
        "truncation_rejection_rate_tpr": round(tpr, 4),
        "false_rejection_rate_fpr": round(fpr, 4),
        "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
        "meets_A6_rejection_criteria": (tpr >= 0.90 and fpr <= 0.10),
    }


class BenchmarkAggregator:
    """Aggregates metrics over multiple test sequences for Method A vs Method B comparison."""

    def __init__(self):
        self.method_a_results: List[Dict] = []
        self.method_b_results: List[Dict] = []

    def add_result(self, method: str, eval_data: Dict) -> None:
        if method == "METHOD_A_MULTIVIEW":
            self.method_a_results.append(eval_data)
        else:
            self.method_b_results.append(eval_data)

    def summarize(self) -> Dict:
        """Produces full summary and comparative analysis between Method A and Method B."""
        def calc_stats(results: List[Dict]) -> Dict:
            if not results:
                return {}
            maes = [r["mae_cm"] for r in results]
            vw_withins = [1.0 if r["within_15_pct"] else 0.0 for r in results]
            ious = [r["iou_3d"] for r in results]
            lats = [r["latency_seconds"] for r in results]
            accepted = [1.0 if r.get("status") == "ACCEPTED" else 0.0 for r in results]

            return {
                "total_sequences": len(results),
                "acceptance_rate_pct": round(float(np.mean(accepted) * 100.0), 2),
                "mean_mae_cm": round(float(np.mean(maes)), 3),
                "std_mae_cm": round(float(np.std(maes)), 3),
                "volumetric_weight_within_15pct_fraction": round(float(np.mean(vw_withins)), 4),
                "volumetric_weight_within_15pct_pct": round(float(np.mean(vw_withins) * 100.0), 2),
                "mean_3d_iou": round(float(np.mean(ious)), 4),
                "mean_latency_seconds": round(float(np.mean(lats)), 4),
                "p95_latency_seconds": round(float(np.percentile(lats, 95)), 4) if len(lats) > 0 else 0.0,
                "max_latency_seconds": round(float(np.max(lats)), 4),
            }

        stats_a = calc_stats(self.method_a_results)
        stats_b = calc_stats(self.method_b_results)

        # Evaluate acceptance criteria
        criteria_summary = {
            "A2_method_a_mae_le_2_0_cm": bool(stats_a.get("mean_mae_cm", 999.0) <= 2.0),
            "A3_method_a_vol_weight_within_15pct_ge_75pct": bool(stats_a.get("volumetric_weight_within_15pct_fraction", 0.0) >= 0.75),
            "A4_method_a_mean_3d_iou_ge_0_50": bool(stats_a.get("mean_3d_iou", 0.0) >= 0.50),
            "A7_method_a_latency_le_2_0_s": bool(stats_a.get("mean_latency_seconds", 999.0) <= 2.0),
        }

        return {
            "method_a": stats_a,
            "method_b": stats_b,
            "acceptance_criteria": criteria_summary,
            "all_criteria_met": all(criteria_summary.values()),
        }
