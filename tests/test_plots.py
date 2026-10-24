"""The raw points and the two plots of one pair."""
import json

import numpy as np
import pytest

from sameface.measure import measure
from sameface.plots import (drift_histogram, drift_zones, point_table, same_size,
                            zone_medians)
from test_measure import a_patch_moved, oval_masker, still, textured


@pytest.fixture(scope="module")
def drifted():
    before = textured(320, 400)
    return measure(before, textured(320, 400, 1), oval_masker, a_patch_moved(6))


def test_the_table_has_a_row_per_point_and_reads_as_json(drifted):
    table = point_table([drifted, drifted])
    rows = 2 * len(drifted.before_points)
    assert all(len(column) == rows for column in table.values())
    assert set(table) == {"pair", "x", "y", "dx_similarity", "dy_similarity",
                          "drift_similarity", "dx_affine", "dy_affine", "drift_affine",
                          "confidence", "part"}
    assert table["pair"][0] == 0 and table["pair"][-1] == 1
    assert json.loads(json.dumps(table)) == table


def test_the_table_carries_the_fits_own_drift(drifted):
    table = point_table([drifted])
    for mode in ("similarity", "affine"):
        assert table[f"drift_{mode}"] == pytest.approx(list(drifted.fits[mode].residuals))
    assert set(table["part"]) <= {"face", "ear"}


def test_zone_medians_put_each_point_in_its_cell():
    xs, ys = np.meshgrid(np.arange(30) + 0.5, np.arange(40) + 0.5)
    points = np.column_stack([xs.ravel(), ys.ravel()])
    drift = (points[:, 1] // 10) * 10 + points[:, 0] // 10     # row * 10 + column
    medians = zone_medians(points, drift, rows=4, cols=3, min_points=5)
    assert medians.shape == (4, 3)
    assert medians == pytest.approx(np.array([[0, 1, 2], [10, 11, 12],
                                              [20, 21, 22], [30, 31, 32]]))


def test_a_cell_with_too_few_points_is_left_empty():
    points = np.array([[1.0, 1.0], [9.0, 9.0], [9.5, 9.5], [9.2, 9.8]])
    medians = zone_medians(points, np.ones(4), rows=2, cols=2, min_points=2)
    assert np.isnan(medians[0, 0]) and medians[1, 1] == 1.0


def test_the_histogram_is_an_rgb_image_of_the_asked_size(drifted):
    image = drift_histogram(drifted, 0.05, size=(900, 450))
    assert image.shape == (450, 900, 3) and image.dtype == np.uint8


def test_no_drift_still_draws():
    before = textured(320, 400)
    drift = measure(before, before.copy(), oval_masker, still)
    drift_histogram(drift, 0.05)
    drift_zones(drift, "similarity", 0.05)


def test_the_moved_patch_is_the_worst_zone(drifted):
    # The patch sits in the middle of the face: the middle column, between
    # the second and third of four rows.
    medians = zone_medians(drifted.before_points, drifted.fits["similarity"].residuals,
                           rows=4, cols=3, min_points=5)
    row, col = np.unravel_index(np.nanargmax(medians), medians.shape)
    assert col == 1 and row in (1, 2)


def test_the_zones_picture_is_a_fixed_height(drifted):
    image = drift_zones(drifted, "similarity", 0.05)
    assert image.shape[0] == 1000 and image.shape[2] == 3 and image.dtype == np.uint8


def test_same_size_pads_to_the_largest():
    small, large = np.zeros((10, 20, 3), np.uint8), np.zeros((30, 15, 3), np.uint8)
    padded = same_size([small, large])
    assert [image.shape for image in padded] == [(30, 20, 3), (30, 20, 3)]
