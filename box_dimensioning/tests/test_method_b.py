"""Automated tests for Method B Single-view Metrology and Comparative Evaluation (A5)."""

import os
import pytest
import numpy as np

from box_dimensioning.dataset.objectron_parser import load_sequence
from box_dimensioning.pipeline.dimensioner import BoxDimensioner


def test_method_b_pipeline_and_comparison():
    """Verify Method B executes, meets schema requirements, and compares with Method A (A5)."""
    seq_dir = "data/objectron/cereal_box/batch-26/22"
    assert os.path.isdir(seq_dir), f"Real Objectron sequence required at {seq_dir}"

    seq = load_sequence(seq_dir, "cereal_box/batch-26/22")
    dimensioner = BoxDimensioner()

    res_b = dimensioner.dimension_sequence(seq, method="METHOD_B_SINGLEVIEW")
    res_a = dimensioner.dimension_sequence(seq, method="METHOD_A_MULTIVIEW")

    # Verify JSON schema compliance
    assert "dimensions_cm" in res_b
    assert "volumetric_weight_kg" in res_b
    assert "status" in res_b
    assert "quality_flags" in res_b
    assert res_b["method"] == "METHOD_B_SINGLEVIEW"

    dims_b = res_b["dimensions_cm"]
    assert dims_b["length"] > 0.0
    assert dims_b["width"] > 0.0
    assert dims_b["height"] > 0.0
    assert res_b["volumetric_weight_kg"] > 0.0

    print(f"\n[Test Method B Comparison (A5)]")
    print(f"  Method A Latency: {res_a['latency_seconds']:.4f} s")
    print(f"  Method B Latency: {res_b['latency_seconds']:.4f} s")
    print(f"  Method A Dims: {res_a['dimensions_cm']}")
    print(f"  Method B Dims: {res_b['dimensions_cm']}")

    # Method B is expected to run fast on a single frame (< 0.5 s)
    assert res_b["latency_seconds"] <= 0.5
