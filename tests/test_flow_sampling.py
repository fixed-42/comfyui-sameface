"""The part of a flow measurement that needs no model: where to look, how the
two sides are brought into one frame, and which sampled points are kept.

It lives outside the RAFT worker so the app can import it too. The panel that
shows "what RAFT was given" renders its crops through `crop_to_working` - the
same function the worker used - rather than through a lookalike, and a
lookalike is exactly how a picture and a measurement come to disagree.

No GPU and no weights here. The flows are made by hand, so each test knows
what the answer has to be.
"""
import numpy as np
import pytest

from sameface.flow_sampling import (
    BOX_PAD, crop_points, crop_to_working, face_box, outcomes, region_contour,
    sample, tags, to_image_pixels, to_working_pixels,
)

SIZE, STEP, LIMIT = 64, 16, 1.5


def flows(dx=0.0, dy=0.0, back=None):
    """A forward flow of (dx, dy) everywhere, and the reverse that undoes it."""
    forward = np.zeros((2, SIZE, SIZE), np.float32)
    forward[0], forward[1] = dx, dy
    return forward, (-forward if back is None else back)


# ---------- where to look ----------


def test_the_box_is_padded_on_every_side_by_a_share_of_its_own_extent():
    box = face_box([[100, 200], [300, 600]], width=1000, height=1000)

    assert box == pytest.approx((100 - 200 * BOX_PAD, 200 - 400 * BOX_PAD,
                                 300 + 200 * BOX_PAD, 600 + 400 * BOX_PAD))


def test_the_box_stops_at_the_image_edge():
    """A face that fills the image has nowhere to pad into. The box is cut,
    and cut differently on a side that sits differently in its image - which
    is one way the two crops of a pair come to differ in proportion."""
    box = face_box([[10, 10], [990, 700]], width=1000, height=720)

    assert box == pytest.approx((0.0, 0.0, 1000.0, 720.0))


# ---------- one frame for both sides ----------


def test_the_crop_is_the_working_size_whatever_the_box():
    image = np.zeros((300, 500, 3), np.uint8)

    frame, _ = crop_to_working(image, (50.0, 20.0, 450.0, 280.0), SIZE)

    assert frame.shape == (SIZE, SIZE, 3)


def test_working_pixels_map_back_to_the_image_pixels_they_came_from():
    """The inverse is the whole reason cropping is legitimate: discard it and
    the global similarity being measured goes with it."""
    image = np.zeros((300, 500, 3), np.uint8)
    box = (50.0, 20.0, 450.0, 280.0)
    _, transform = crop_to_working(image, box, SIZE)

    corners = np.array([[0.0, 0.0], [SIZE, SIZE]])
    assert np.allclose(to_image_pixels(corners, transform), [[50, 20], [450, 280]])
    assert np.allclose(
        to_working_pixels(to_image_pixels(corners, transform), transform), corners)


def test_the_crop_shows_what_is_in_the_box():
    image = np.zeros((100, 100, 3), np.uint8)
    image[40:60, 40:60] = 255           # a white square, exactly the box

    frame, _ = crop_to_working(image, (40.0, 40.0, 60.0, 60.0), SIZE)

    # Away from the rim, where resampling blends in the black just outside.
    assert frame[8:-8, 8:-8].min() == 255


# ---------- which points are kept ----------


def test_flow_that_returns_where_it_started_is_all_kept():
    forward, backward = flows(dx=2.0, dy=-1.0)

    sampled = sample(forward, backward, STEP, LIMIT)

    assert sampled.keep.all()
    assert np.allclose(sampled.landed - sampled.origin, [2.0, -1.0])
    assert np.allclose(sampled.error, 0.0, atol=1e-5)


def test_flow_that_does_not_come_back_is_dropped_as_inconsistent():
    forward, _ = flows(dx=2.0)
    # The reverse flow says nothing moved, so every point returns 2px away.
    sampled = sample(forward, np.zeros_like(forward), STEP, LIMIT)

    assert not sampled.keep.any()
    assert set(outcomes(sampled)) == {"dropped_inconsistent"}


def test_a_dropped_point_that_left_the_frame_is_labelled_as_having_left_it():
    """Why it was dropped matters: a point that ran off the crop says the two
    sides were not brought close enough, while one that failed the round trip
    in the middle of the face says the texture could not be matched."""
    forward, _ = flows(dx=float(SIZE))            # everything lands off the right
    sampled = sample(forward, np.zeros_like(forward), STEP, LIMIT)

    assert sampled.outside.all()
    assert set(outcomes(sampled)) == {"dropped_outside"}


def test_a_point_that_left_the_frame_but_passed_is_kept_and_says_so():
    """The reverse flow is read at the frame edge for such a point, and can
    agree by accident. It is kept - changing that would change every stored
    number - but it is labelled, so the question can be asked later."""
    forward, backward = flows(dx=float(SIZE))     # lands off the frame...
    sampled = sample(forward, backward, STEP, LIMIT)    # ...and "returns"

    assert sampled.keep.all() and sampled.outside.all()
    assert set(outcomes(sampled)) == {"kept_outside"}


def test_labelling_changes_nothing_about_which_points_are_kept():
    """The old rule, transcribed, as an independent oracle: keep where the
    round-trip error is within the limit, reading the reverse flow at the
    nearest pixel, clamped to the frame."""
    rng = np.random.default_rng(7)
    forward = rng.normal(0, 3, (2, SIZE, SIZE)).astype(np.float32)
    backward = (-forward + rng.normal(0, 1, (2, SIZE, SIZE))).astype(np.float32)

    step = 4                        # a finer grid, so both outcomes occur
    sampled = sample(forward, backward, step, LIMIT)

    half = step // 2
    coords = np.arange(half, SIZE - half, step)
    xs, ys = (grid.ravel() for grid in np.meshgrid(coords, coords))
    landed = np.column_stack([xs + forward[0, ys, xs], ys + forward[1, ys, xs]])
    bx = np.clip(np.rint(landed[:, 0]).astype(int), 0, SIZE - 1)
    by = np.clip(np.rint(landed[:, 1]).astype(int), 0, SIZE - 1)
    returned = landed + np.column_stack([backward[0, by, bx], backward[1, by, bx]])
    expected = np.hypot(*(returned - np.column_stack([xs, ys])).T) <= LIMIT

    assert 0 < expected.sum() < len(expected)       # the oracle is not trivial
    assert (sampled.keep == expected).all()


# ---------- the whole crop, and which part each point is on ----------
#
# The flow is read at every grid point in the crop, once. Which of those feed
# a fit is chosen afterwards, so each point says where it sits - on the face,
# on an ear, or outside the mask - instead of being left out for it.


def contour(points):
    return np.array(points, np.float32).reshape(-1, 1, 2)


LEFT_HALF = [[0, 0], [SIZE / 2, 0], [SIZE / 2, SIZE], [0, SIZE]]


def test_every_grid_point_in_the_crop_is_sampled():
    forward, backward = flows()

    sampled = sample(forward, backward, STEP, LIMIT)

    # 8, 24 and 40 each way: the grid stops half a step short of the far edge.
    assert len(sampled.origin) == 3 * 3
    assert sorted(set(sampled.origin[:, 0].tolist())) == [8, 24, 40]


def test_a_point_is_tagged_by_the_region_it_sits_in():
    forward, backward = flows()
    origin = sample(forward, backward, STEP, LIMIT).origin

    found = tags(origin, contour(LEFT_HALF))

    assert set(found) == {"face", "outside"}
    assert all((tag == "face") == (x <= SIZE / 2) for tag, (x, _) in zip(found, origin))


def test_the_part_of_the_region_that_is_not_face_is_ear():
    """The region is face and ears joined. The face alone is the smaller
    outline inside it; what the joining added is the ear's."""
    forward, backward = flows()
    origin = sample(forward, backward, STEP, LIMIT).origin
    top_left = [[0, 0], [SIZE / 2, 0], [SIZE / 2, SIZE / 2], [0, SIZE / 2]]

    found = tags(origin, contour(LEFT_HALF), face=contour(top_left))

    for tag, (x, y) in zip(found, origin):
        assert tag == ("outside" if x > SIZE / 2 else "face" if y <= SIZE / 2 else "ear")


def test_an_ear_standing_apart_from_the_region_is_tagged_too():
    forward, backward = flows()
    origin = sample(forward, backward, STEP, LIMIT).origin
    apart = [[36, 0], [48, 0], [48, STEP], [36, STEP]]

    found = tags(origin, contour(LEFT_HALF), face=contour(LEFT_HALF),
                 ears=[contour(apart)])

    assert [tuple(point) for tag, point in zip(found, origin) if tag == "ear"] == [
        (40, 8)]


def test_with_no_region_the_whole_crop_is_the_face():
    """A box run asks about the whole square."""
    forward, backward = flows()
    origin = sample(forward, backward, STEP, LIMIT).origin

    assert set(tags(origin)) == {"face"}


def test_the_crop_takes_in_an_ear_that_stands_apart():
    """The crop is the face and any ears found, whichever of them feed the
    fit - otherwise ears in and ears out would be different crops, different
    grids and different runs. A joined ear is already inside the region."""
    region = [[10.0, 10.0], [50.0, 10.0], [50.0, 60.0], [10.0, 60.0]]
    parts = {"face": region, "ears": [
        {"points": [[48.0, 30.0], [55.0, 30.0], [55.0, 40.0]], "joined": True},
        {"points": [[70.0, 30.0], [80.0, 30.0], [80.0, 45.0]], "joined": False}]}

    points = crop_points(region, parts)

    assert points.min(axis=0).tolist() == [10.0, 10.0]
    assert points.max(axis=0).tolist() == [80.0, 60.0]
    assert crop_points(region, None).tolist() == region


def test_a_landmark_cloud_is_hulled_and_an_outline_is_taken_as_given():
    """`convex` is the whole difference between a hull and a polygon run."""
    transform = (1.0, 1.0, 0.0, 0.0)
    notched = [[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]]     # a concave outline

    assert len(region_contour(notched, transform, convex=True)) == 4
    assert len(region_contour(notched, transform, convex=False)) == 5
