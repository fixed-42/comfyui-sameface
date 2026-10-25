"""The noise floor's own arithmetic: how null pairs are made and how a floor
is declared from what they measure."""
import cv2
import numpy as np
import pytest

from sameface.calibration import declared_floor, null_pair


def test_the_first_floor_follows_from_its_null_median():
    assert declared_floor([0.025]) == 0.05


def test_the_two_pass_floor_follows_from_its_null_median():
    assert declared_floor([0.0030, 0.0036, 0.0070]) == 0.01


def test_a_floor_rounds_up_to_the_next_step():
    assert declared_floor([0.06]) == 0.2         # 0.12 -> 0.2
    assert declared_floor([0.011]) == 0.05       # 0.022 -> 0.05
    assert declared_floor([0.5]) == 1.0


def test_no_drift_at_all_is_an_error():
    with pytest.raises(ValueError, match="no drift"):
        declared_floor([0.0, 0.0])


def test_a_null_pair_is_one_image_at_two_sizes_keeping_its_shape():
    rgb = np.random.default_rng(0).integers(0, 255, (768, 1024, 3), np.uint8)

    small, large = null_pair(rgb)

    assert small.shape == (384, 512, 3)
    assert large.shape == (1060, 1414, 3)


def test_a_null_pair_passes_through_a_jpeg_unless_asked_not_to():
    # Smooth, as a photo is: pure noise is the one thing a JPEG cannot keep.
    noise = np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8)
    rgb = cv2.resize(noise, (512, 512), interpolation=cv2.INTER_CUBIC)

    small, _ = null_pair(rgb)
    clean, _ = null_pair(rgb, quality=None)

    assert np.array_equal(clean, rgb)
    assert not np.array_equal(small, rgb)
    assert np.abs(small.astype(int) - rgb.astype(int)).mean() < 3
