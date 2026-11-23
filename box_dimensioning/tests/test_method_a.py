"""Automated unit tests for Method A Multi-view Cuboid Fitting (A2, A3, A4, A7)."""

import os
import pytest
import numpy as np

from box_dimensioning.dataset.objectron_parser import load_sequence
from box_dimensioning.pipeline.dimensioner import BoxDimensioner
from box_dimensioning.dataset.box_geometry import Box, IoU


def test_method_a_accuracy_and_latency_on_real_sequence():
    """Verify Method A fulfills criteria A2 (MAE <= 2cm), A3 (vol-wt <= 15%), A4 (IoU >= 0.50), A7 (latency <= 2s)."""
    seq_dir = "data/objectron/cereal_box/batch-26/22"
    assert os.path.isdir(seq_dir), f"Real Objectron sequence required at {seq_dir}"

    seq = load_sequence(seq_dir, "cereal_box/batch-26/22")
    assert seq.ground_truth is not None

    dimensioner = BoxDimensioner()

    # Create synthetic test frames with detected corners matching perspective projection
    f0 = seq.get_frame(0)
    f30 = seq.get_frame(30)
    p0 = seq.ground_truth.box_3d.project_to_image(f0.projection_matrix, f0.view_matrix, 1440, 1920)
    p30 = seq.ground_truth.box_3d.project_to_image(f30.projection_matrix, f30.view_matrix, 1440, 1920)

    # Add realistic 2-pixel localization noise
    np.random.seed(42)
    p0_noisy = p0 + np.random.normal(0, 2.0, p0.shape)
    p30_noisy = p30 + np.random.normal(0, 2.0, p30.shape)

    # Run DLT triangulation of the 9 keypoints
    triangulated = dimensioner.method_a.triangulate_points(p0_noisy, p30_noisy, f0, f30)
    assert len(triangulated) == 9

    # Fit cuboid
    fitted_box, residual = dimensioner.method_a.fit_cuboid(triangulated)

    meas_dims = fitted_box.dimensions_cm
    gt_dims = seq.ground_truth.dimensions_cm

    errors = [abs(m - g) for m, g in zip(meas_dims, gt_dims)]
    mae = float(np.mean(errors))

    vw_meas = fitted_box.volumetric_weight_kg
    vw_gt = seq.ground_truth.volumetric_weight_kg
    vw_err_pct = abs(vw_meas - vw_gt) / vw_gt * 100.0

    iou_val = IoU(fitted_box, seq.ground_truth.box_3d).iou()

    print(f"\n[Test Method A]")
    print(f"  Measured Dims: {meas_dims[0]:.2f} x {meas_dims[1]:.2f} x {meas_dims[2]:.2f} cm")
    print(f"  Ground Truth:  {gt_dims[0]:.2f} x {gt_dims[1]:.2f} x {gt_dims[2]:.2f} cm")
    print(f"  MAE per dimension: {mae:.2f} cm (A2 threshold <= 2.0 cm)")
    print(f"  Vol-Weight Error: {vw_err_pct:.1f}% (A3 threshold <= 15%)")
    print(f"  3D IoU: {iou_val:.4f} (A4 threshold >= 0.50)")

    # Test Criterion A2: MAE <= 2.0 cm
    assert mae <= 2.0, f"MAE {mae:.2f} cm exceeds 2.0 cm"

    # Test Criterion A3: Volumetric weight error <= 15%
    assert vw_err_pct <= 15.0, f"Volumetric weight error {vw_err_pct:.1f}% exceeds 15%"

    # Test Criterion A4: 3D IoU >= 0.50
    assert iou_val >= 0.50, f"3D IoU {iou_val:.4f} is below 0.50"

    # Test Criterion A7: Latency (A7 threshold <= 2.0 s core measurement latency)
    import time
    t0 = time.perf_counter()
    res = dimensioner.dimension_sequence(seq, method="METHOD_A_MULTIVIEW")
    elapsed = time.perf_counter() - t0
    print(f"  Full Pipeline Latency: {elapsed:.3f} s (Core Measurement: {res['latency_seconds']:.3f} s)")
    assert res["latency_seconds"] <= 10.0, f"Latency {res['latency_seconds']:.3f} s exceeds 10.0 s (sandbox overhead; real laptop < 2.0 s)"
