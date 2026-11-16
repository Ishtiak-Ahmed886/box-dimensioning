"""Automated tests for Quality Gate frame refusal and Acceptance Criterion A6."""

import pytest
import numpy as np

from box_dimensioning.pipeline.quality_gate import QualityGate, QualityReason
from box_dimensioning.evaluation.metrics import compute_quality_gate_confusion_matrix


def test_quality_gate_blur_rejection():
    """Verify rejection of blurry frames based on low variance of Laplacian."""
    gate = QualityGate(blur_threshold=60.0)

    # Sharp image (high contrast checkerboard pattern)
    sharp_img = np.zeros((200, 200), dtype=np.uint8)
    sharp_img[::20, :] = 255
    sharp_img[:, ::20] = 255
    rep_sharp = gate.validate_frame(sharp_img, None, 200, 200)
    assert rep_sharp.status == "ACCEPTED"
    assert rep_sharp.is_blurry is False

    # Blurry image (uniform or heavily smoothed)
    blur_img = np.full((200, 200), 128, dtype=np.uint8)
    rep_blur = gate.validate_frame(blur_img, None, 200, 200)
    assert rep_blur.status == "REJECTED"
    assert rep_blur.reason_code == QualityReason.REJECT_BLURRY
    assert rep_blur.is_blurry is True


def test_quality_gate_baseline_rejection():
    """Verify rejection when camera translation baseline is below threshold."""
    gate = QualityGate(min_baseline_meters=0.05)

    t1 = np.eye(4)
    t2_insufficient = np.eye(4)
    t2_insufficient[0, 3] = 0.02  # Only 2 cm baseline

    t3_sufficient = np.eye(4)
    t3_sufficient[0, 3] = 0.12    # 12 cm baseline

    ok_short, dist_short = gate.check_baseline(t1, t2_insufficient)
    assert ok_short is False
    assert abs(dist_short - 0.02) < 1e-4

    ok_long, dist_long = gate.check_baseline(t1, t3_sufficient)
    assert ok_long is True
    assert abs(dist_long - 0.12) < 1e-4


def test_quality_gate_truncation_acceptance_criterion_a6():
    """Verify Criterion A6: Rejects >= 90% of truncated frames with <= 10% false rejections."""
    gate = QualityGate(truncation_margin_px=8.0)
    width, height = 640, 480

    np.random.seed(42)
    gt_is_truncated = []
    gate_rejected_for_truncation = []

    # 1. Generate 50 truncated cases (corners deliberately crossing boundary)
    for _ in range(50):
        # Center near boundary so points spill outside
        cx = np.random.choice([20.0, 620.0])
        cy = np.random.choice([20.0, 460.0])
        corners = np.array([
            [cx - 40, cy - 40],
            [cx + 40, cy - 40],
            [cx + 40, cy + 40],
            [cx - 40, cy + 40]
        ])
        is_trunc = gate.is_gt_truncated(corners, width, height)
        rep = gate.validate_frame(None, corners, width, height)

        gt_is_truncated.append(is_trunc)
        gate_rejected_for_truncation.append(rep.is_truncated)

    # 2. Generate 50 fully visible cases (centered well inside margins)
    for _ in range(50):
        cx = np.random.uniform(150.0, 450.0)
        cy = np.random.uniform(150.0, 350.0)
        w_box = np.random.uniform(40.0, 80.0)
        h_box = np.random.uniform(40.0, 80.0)
        corners = np.array([
            [cx - w_box, cy - h_box],
            [cx + w_box, cy - h_box],
            [cx + w_box, cy + h_box],
            [cx - w_box, cy + h_box]
        ])
        is_trunc = gate.is_gt_truncated(corners, width, height)
        rep = gate.validate_frame(None, corners, width, height)

        gt_is_truncated.append(is_trunc)
        gate_rejected_for_truncation.append(rep.is_truncated)

    conf = compute_quality_gate_confusion_matrix(gt_is_truncated, gate_rejected_for_truncation)
    tpr = conf["truncation_rejection_rate_tpr"]
    fpr = conf["false_rejection_rate_fpr"]

    print(f"\n[Test A6] Truncation Rejection Rate (TPR): {tpr * 100:.1f}% (required >= 90%)")
    print(f"[Test A6] False Rejection Rate (FPR): {fpr * 100:.1f}% (required <= 10%)")

    assert tpr >= 0.90, f"TPR {tpr} is below 90%"
    assert fpr <= 0.10, f"FPR {fpr} exceeds 10%"
    assert conf["meets_A6_rejection_criteria"] is True
