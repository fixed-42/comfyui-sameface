"""The smooth warp: moving an after image's features back, part by part.

A similarity or an affine moves the whole image as one sheet. What it leaves
is parts of the face sitting in different places relative to each other, and
no sheet can fix that. This warp moves each part by its own amount.

Every matched point says how far the image is off at its place. A pixel moves
by a smooth blend of what the points near it say - fully at a point, less
with distance, not at all by the time the next points are reached if those
did not move. No triangles and so no creases: the blend has no edges.

Three properties carry the weight, and each can be wrong without looking it:

- **direction** - the after content is brought *to* where the before has it,
  not pushed the same distance the other way;
- **it stops** - outside the points the image is left as the flat fit put it,
  or hair and background are dragged along;
- **it gives way** - a lone point that moved against still neighbours is a
  bad match more often than a feature, and is not followed all the way.

No GPU and no photographs: the points and the pictures are made by hand.
"""
import numpy as np
import pytest

from sameface.smooth_warp import field, render_repaired

SIZE = 200
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def grid(lo=60, hi=140, step=10):
    return np.array([[x, y] for y in range(lo, hi + 1, step)
                     for x in range(lo, hi + 1, step)], float)


def at(found, x, y):
    return found[int(y), int(x)]


# ---------- the field: how far each pixel is moved ----------


def test_where_nothing_moved_nothing_is_moved():
    found = field(grid(), np.zeros((len(grid()), 2)), (SIZE, SIZE))

    assert np.abs(found).max() < 1e-6


def test_one_point_moves_and_its_neighbours_hold_still():
    """The Liquify picture: full at the point, fading, gone by the anchors."""
    points = grid()
    offsets = np.zeros_like(points)
    centre = int(np.flatnonzero((points == [100, 100]).all(axis=1))[0])
    offsets[centre] = [4.0, 0.0]

    found = field(points, offsets, (SIZE, SIZE), smoothing=0.0)

    here = at(found, 100, 100)[0]
    half = at(found, 105, 100)[0]
    next_door = at(found, 110, 100)[0]
    assert here == pytest.approx(4.0, abs=0.3)
    assert 0.5 < half < here                       # less with distance
    assert abs(next_door) < 0.3                    # the anchor holds
    assert abs(at(found, 100, 100)[1]) < 0.2       # and only the way it was told


def test_points_that_all_move_together_carry_everything_between_them():
    points = grid()
    offsets = np.tile([3.0, -2.0], (len(points), 1))

    found = field(points, offsets, (SIZE, SIZE))

    for x, y in ((100, 100), (75, 125), (95, 65)):
        assert at(found, x, y) == pytest.approx([3.0, -2.0], abs=0.15)


def test_it_stops_outside_the_points():
    """The face is moved; what is beyond it is left as the flat fit put it."""
    points = grid()
    offsets = np.tile([3.0, -2.0], (len(points), 1))

    found = field(points, offsets, (SIZE, SIZE))

    for x, y in ((5, 5), (195, 100), (100, 190)):
        assert np.abs(at(found, x, y)).max() == 0.0
    # ...and gets there gradually. Within a spacing of the last point it is
    # still whole; a little further out it is part of the way.
    assert at(found, 148, 100)[0] == pytest.approx(3.0, abs=0.15)
    assert 0.5 < at(found, 158, 100)[0] < 2.9


def test_a_lone_spike_is_not_followed_all_the_way():
    """One point moving against still neighbours is usually a bad match.
    With smoothing the warp gives way to it; without, it passes through."""
    points = grid()
    offsets = np.zeros_like(points)
    centre = int(np.flatnonzero((points == [100, 100]).all(axis=1))[0])
    offsets[centre] = [6.0, 0.0]

    exact = at(field(points, offsets, (SIZE, SIZE), smoothing=0.0), 100, 100)[0]
    eased = at(field(points, offsets, (SIZE, SIZE)), 100, 100)[0]

    assert exact == pytest.approx(6.0, abs=0.4)
    assert 0.5 < eased < exact * 0.8


def test_smoothing_leaves_a_broad_movement_alone():
    """What it gives way to is the spike, not the feature: points moving
    together are followed with smoothing on."""
    points = grid()
    offsets = np.zeros_like(points)
    moved = (np.abs(points - [100, 100]) <= 20).all(axis=1)      # a 5 x 5 patch
    offsets[moved] = [4.0, 0.0]

    found = field(points, offsets, (SIZE, SIZE))

    assert at(found, 100, 100)[0] == pytest.approx(4.0, abs=0.5)


# ---------- the picture: where the content ends up ----------


def dot_image(x, y, size=SIZE):
    image = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[:size, :size]
    image[(xx - x) ** 2 + (yy - y) ** 2 <= 9] = 255
    return image


def centre_of(rendered):
    ys, xs = np.nonzero(rendered[..., 0] > 128)
    return xs.mean(), ys.mean()


def a_feature_out_of_place(shift):
    """Before has every point where after has it, except a patch around
    (100, 100): what after shows there belongs `shift` further on."""
    after = grid()
    before = after.copy()
    before[(np.abs(after - [100, 100]) <= 20).all(axis=1)] += shift
    return after, before


def test_the_after_content_is_brought_to_where_the_before_has_it():
    """The direction. After shows a dot at (100, 100); the points say that
    part of the face belongs 5 px to the right. Repaired, it is there - not
    5 px to the left, which is what applying the field backwards gives."""
    after, before = a_feature_out_of_place([5.0, 0.0])

    repaired = render_repaired(dot_image(100, 100), after, before, IDENTITY,
                               (SIZE, SIZE))

    x, y = centre_of(repaired)
    assert (x, y) == pytest.approx((105.0, 100.0), abs=0.7)


def test_at_no_strength_it_is_the_flat_fit_and_nothing_else():
    after, before = a_feature_out_of_place([5.0, 0.0])

    repaired = render_repaired(dot_image(100, 100), after, before, IDENTITY,
                               (SIZE, SIZE), strength=0.0)

    assert centre_of(repaired) == pytest.approx((100.0, 100.0), abs=0.1)


def test_strength_is_how_far_along_the_way_it_goes():
    after, before = a_feature_out_of_place([6.0, 0.0])

    repaired = render_repaired(dot_image(100, 100), after, before, IDENTITY,
                               (SIZE, SIZE), strength=0.5)

    assert centre_of(repaired)[0] == pytest.approx(103.0, abs=0.7)


def test_the_flat_fit_is_applied_underneath():
    """A similarity that doubles the after image, and a warp on top of it:
    the dot lands where the fit puts it, moved on by what the points add."""
    double = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    after = grid(30, 70, 5)
    before = after * 2.0
    before[(np.abs(after - [50, 50]) <= 10).all(axis=1)] += [4.0, 0.0]

    repaired = render_repaired(dot_image(50, 50), after, before, double,
                               (SIZE, SIZE))

    assert centre_of(repaired) == pytest.approx((104.0, 100.0), abs=1.0)


def test_content_away_from_the_points_is_where_the_flat_fit_put_it():
    after, before = a_feature_out_of_place([5.0, 0.0])

    repaired = render_repaired(dot_image(20, 180), after, before, IDENTITY,
                               (SIZE, SIZE))

    assert centre_of(repaired) == pytest.approx((20.0, 180.0), abs=0.1)


def test_moved_is_where_the_render_puts_an_after_point():
    """The render answers which after point lands at each pixel; `moved`
    answers where an after point lands. A dot moved by the warp is found
    where `moved` says."""
    from sameface.smooth_warp import Warp
    after, before = a_feature_out_of_place([5.0, 0.0])
    warp = Warp(after, before, IDENTITY, (SIZE, SIZE))

    for strength in (0.0, 0.5, 1.0):
        landed = warp.moved([[100.0, 100.0]], strength)[0]
        assert tuple(landed) == pytest.approx(
            centre_of(warp.render(dot_image(100, 100), strength)), abs=0.7)
