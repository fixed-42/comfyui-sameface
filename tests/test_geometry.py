"""The flat fits and the figures they leave, on points made by hand."""
import numpy as np
import pytest

from sameface import geometry


def grid(low=0.0, high=100.0, count=11):
    xs = np.linspace(low, high, count)
    return np.array([[x, y] for y in xs for x in xs])


def similarity(points, scale=1.0, degrees=0.0, shift=(0.0, 0.0)):
    theta = np.radians(degrees)
    turn = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return points @ turn.T + shift


def test_a_similarity_is_recovered_exactly_and_leaves_nothing():
    source = grid()
    target = similarity(source, 1.3, 12.0, (40.0, -7.0))

    matrix = geometry.fit(source, target, "similarity")
    left = geometry.residuals(geometry.apply(matrix, source), target)

    assert left.max() == pytest.approx(0.0, abs=1e-9)
    described = geometry.describe(matrix, source, target)
    assert described["scale"] == pytest.approx(1.3)
    assert described["rotation_deg"] == pytest.approx(12.0)


def test_a_stretch_is_left_by_a_similarity_and_taken_out_by_an_affine():
    source = grid()
    target = source * [1.1, 0.9]

    by_similarity = geometry.residuals(
        geometry.apply(geometry.fit(source, target, "similarity"), source), target)
    affine = geometry.fit(source, target, "affine")
    by_affine = geometry.residuals(geometry.apply(affine, source), target)

    assert np.median(by_similarity) > 1.0
    assert by_affine.max() == pytest.approx(0.0, abs=1e-9)
    described = geometry.describe(affine, source, target)
    assert (described["stretch_x"], described["stretch_y"]) == pytest.approx((1.1, 0.9))
    assert described["shear_deg"] == pytest.approx(0.0, abs=1e-9)


def test_a_mirror_is_never_fitted():
    """The best similarity onto a mirrored face is a poor fit, not a mirror."""
    source = grid()
    source[:, 0] **= 1.2                      # lopsided, so a turn can do something
    target = source * [-1.0, 1.0]

    matrix = geometry.fit(source, target, "similarity")

    assert np.linalg.det(matrix[:, :2]) >= 0


def test_residuals_are_a_share_of_the_face_whatever_its_size():
    small = grid(0, 100)
    moved = small.copy()
    moved[0] += [3.0, 0.0]

    left_small = geometry.residuals(moved, small)
    left_large = geometry.residuals(moved * 10, small * 10)

    assert left_small == pytest.approx(left_large)


def test_too_few_points_is_refused():
    with pytest.raises(ValueError, match="affine needs at least 3"):
        geometry.fit([[0, 0], [1, 1]], [[0, 0], [1, 1]], "affine")


def test_into_frame_covers_and_centres():
    # A 200x100 image over a 100x100 frame: cover fills the frame, so the
    # scale is 1.0 and 50 px hang off each side.
    points = np.array([[50.0, 0.0], [150.0, 100.0]])
    assert geometry.into_frame(points, (200, 100), (100, 100)) == pytest.approx(
        np.array([[0.0, 0.0], [100.0, 100.0]]))
    # Its own size is the identity.
    assert geometry.into_frame(points, (200, 100), (200, 100)) == pytest.approx(points)
    # Twice as large is twice as far.
    assert geometry.into_frame(points, (200, 100), (400, 200)) == pytest.approx(points * 2)
