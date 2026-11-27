"""Unit tests for BoxDimensioner method routing (Fix 1: F4/A5)."""

import numpy as np
import pytest

from box_dimensioning.dataset.objectron_parser import load_sequence
from box_dimensioning.pipeline.dimensioner import BoxDimensioner, DimensioningMethod, normalize_method


def test_normalize_method_aliases_and_invalid():
    """Verify case-insensitive method normalization and ValueError on unknown values."""
    assert normalize_method("A") == DimensioningMethod.METHOD_A
    assert normalize_method("a") == DimensioningMethod.METHOD_A
    assert normalize_method("METHOD_A_MULTIVIEW") == DimensioningMethod.METHOD_A
    assert normalize_method("B") == DimensioningMethod.METHOD_B
    assert normalize_method("b") == DimensioningMethod.METHOD_B
    assert normalize_method("method_b_singleview") == DimensioningMethod.METHOD_B

    with pytest.raises(ValueError, match="Unknown dimensioning method"):
        normalize_method("C")
    with pytest.raises(ValueError, match="Unknown dimensioning method"):
        normalize_method("METHOD_X")


def test_dimension_sequence_routes_method_b_correctly():
    """Prove method='B' executes MethodBSingleView rather than falling back to Method A."""
    seq = load_sequence("data/objectron/cereal_box/batch-26/22", "cereal_box/batch-26/22")
    # Pass 2 synthetic frames to keep test fast while exercising full Method B routing
    img = seq.get_image(0)
    if img is None:
        img = np.full((640, 480, 3), 128, dtype=np.uint8)
    images = {0: img, 15: img}

    dimensioner = BoxDimensioner()
    res_b = dimensioner.dimension_sequence(seq, images=images, method="B")
    assert res_b["method"] == "METHOD_B_SINGLEVIEW"

    res_b_lower = dimensioner.dimension_sequence(seq, images=images, method="method_b_singleview")
    assert res_b_lower["method"] == "METHOD_B_SINGLEVIEW"

    with pytest.raises(ValueError):
        dimensioner.dimension_sequence(seq, images=images, method="INVALID")
