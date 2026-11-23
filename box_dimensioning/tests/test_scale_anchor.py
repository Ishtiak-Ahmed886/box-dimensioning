"""
Test suite for scale anchor and dimensioning consistency.
Verifies that:
1. When raw dimensions are provided in cm without reference, no double metric scaling occurs.
2. Dimensions and volumetric weight stay strictly within realistic courier parcel bounds.
3. Reference object scaling with known pixel dimensions correctly yields real-world measurements.
"""
import numpy as np
import pytest
from box_dimensioning.pipeline.scale_anchor import (
    scale_dimensions_to_cm,
    REFERENCE_OBJECTS,
    DEFAULT_REFERENCE,
)
from box_dimensioning.pipeline.custom_video_processor import compute_volumetric_weight


def test_scale_dimensions_cm_preservation():
    """Verifies that dimensions in cm are not multiplied by 100 again (the double scaling bug)."""
    raw_dims_cm = (28.0, 18.0, 12.0)
    length, width, height = scale_dimensions_to_cm(
        raw_dims=raw_dims_cm,
        scale_mm_per_px=None,
        raw_unit="cm"
    )
    assert length == pytest.approx(28.0, abs=0.1)
    assert width == pytest.approx(18.0, abs=0.1)
    assert height == pytest.approx(12.0, abs=0.1)

    vw = compute_volumetric_weight(length, width, height)
    # 28 * 18 * 12 / 5000 = 1.2096 kg
    assert vw == pytest.approx(1.21, abs=0.05)


def test_scale_dimensions_reference_card_calibration():
    """Verifies that reference object scale correctly computes parcel size from pixel box."""
    # Suppose credit card (85.6mm) is 171.2px -> 0.5 mm/px
    scale_mm_per_px = 0.5
    pixel_dims = (600.0, 400.0, 200.0)  # 30cm x 20cm x 10cm
    raw_dims = (30.0, 20.0, 10.0)

    length, width, height = scale_dimensions_to_cm(
        raw_dims=raw_dims,
        scale_mm_per_px=scale_mm_per_px,
        pixel_dims=pixel_dims,
        raw_unit="cm"
    )
    # 600 * 0.5 / 10 = 30 cm
    # 400 * 0.5 / 10 = 20 cm
    # 200 * 0.5 / 10 = 10 cm
    assert length == pytest.approx(30.0, abs=0.1)
    assert width == pytest.approx(20.0, abs=0.1)
    assert height == pytest.approx(10.0, abs=0.1)

    vw = compute_volumetric_weight(length, width, height)
    # 30 * 20 * 10 / 5000 = 1.2 kg
    assert vw == pytest.approx(1.20, abs=0.01)


def test_scale_dimensions_realistic_clamping():
    """Verifies that extreme inputs are clamped to realistic courier parcel bounds."""
    # Oversized box
    huge_dims = (500.0, 400.0, 300.0)
    l, w, h = scale_dimensions_to_cm(raw_dims=huge_dims, scale_mm_per_px=None, raw_unit="cm")
    assert l <= 120.0
    assert w <= 100.0
    assert h <= 80.0

    # Tiny box
    tiny_dims = (1.0, 0.5, 0.2)
    l, w, h = scale_dimensions_to_cm(raw_dims=tiny_dims, scale_mm_per_px=None, raw_unit="cm")
    assert l >= 5.0
    assert w >= 4.0
    assert h >= 2.0
