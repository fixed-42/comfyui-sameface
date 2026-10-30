"""The known-warp check's own arithmetic: a warp moves a point where it says,
a restyle moves nothing, and a measurement or a repair is scored against the
truth. No models: the points are given, as `measure` would give them."""
import cv2
import numpy as np
import pytest

from sameface.known_warp import (NO_WARP, bump, face_points, left_by_repair, pixel_gap, restyle,
                                 score, tracked_drift, widen)

SIZE = 200


def dot_image(x, y, size=SIZE):
    """A soft white dot on black, its centre at (x, y)."""
    grid_x, grid_y = np.meshgrid(np.arange(size, dtype=float), np.arange(size, dtype=float))
    dot = 255.0 * np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * 4.0 ** 2))
    return np.dstack([dot] * 3).round().astype(np.uint8)


def centre_of(rgb):
    weight = rgb[..., 0].astype(float)
    grid_x, grid_y = np.meshgrid(np.arange(weight.shape[1]), np.arange(weight.shape[0]))
    return (grid_x * weight).sum() / weight.sum(), (grid_y * weight).sum() / weight.sum()


def grid_points(low=40, high=160, step=8):
    xs, ys = np.meshgrid(np.arange(low, high, step, dtype=float),
                         np.arange(low, high, step, dtype=float))
    return np.column_stack([xs.ravel(), ys.ravel()])


def test_a_bump_moves_a_dot_at_its_centre_by_its_vector():
    # Much wider than the dot, so all of the dot moves alike.
    moved = bump((100, 100), 80.0, (6.0, -3.0)).render(dot_image(100, 100))
    assert centre_of(moved) == pytest.approx((106.0, 97.0), abs=0.2)


def test_a_bump_leaves_a_far_dot_where_it_was():
    moved = bump((50, 50), 10.0, (6.0, 0.0)).render(dot_image(150, 150))
    assert centre_of(moved) == pytest.approx((150.0, 150.0), abs=0.1)


def test_widening_moves_a_dot_where_forward_says():
    warp = widen((100, 100), (60, 60), 0.05)
    expected = warp.forward(np.array([[140.0, 100.0]]))[0]

    assert expected == pytest.approx([142.0, 100.0])
    assert centre_of(warp.render(dot_image(140, 100))) == pytest.approx(tuple(expected), abs=0.2)


def test_widening_stops_beyond_its_reach():
    warp = widen((100, 100), (30, 30), 0.05)
    far = np.array([[100.0 + 2.5 * 30, 100.0]])
    assert warp.forward(far) == pytest.approx(far)


def test_no_warp_is_the_same_image():
    image = dot_image(90, 110)
    assert np.array_equal(NO_WARP.render(image), image)


def test_a_restyle_changes_the_pixels_and_moves_nothing():
    image = dot_image(100, 100)
    styled = restyle(image)
    assert np.abs(styled.astype(int) - image).max() > 5
    assert centre_of(styled) == pytest.approx((100.0, 100.0), abs=0.2)


def test_a_perfect_measurement_has_no_error_and_reads_the_whole_move():
    warp = bump((100, 100), 25.0, (5.0, 0.0))
    before = grid_points()

    result = score(warp.forward(before), before, warp)

    assert result.point_error == pytest.approx(0.0, abs=1e-9)
    assert result.read == pytest.approx(1.0)
    for fit in ("similarity", "affine"):
        assert result.measured[fit] == pytest.approx(result.true[fit])


def test_a_measurement_that_saw_half_the_move_reads_half():
    warp = bump((100, 100), 25.0, (5.0, 0.0))
    before = grid_points()

    result = score(before + 0.5 * warp.offset(before), before, warp)

    assert result.read == pytest.approx(0.5)
    assert result.point_error > 0.0


def test_no_warp_has_no_true_drift_and_nothing_to_read():
    before = grid_points()
    result = score(before.copy(), before, NO_WARP)
    assert result.true["similarity"] == pytest.approx(0.0, abs=1e-9)
    assert result.read is None


def test_a_repair_through_true_points_leaves_almost_nothing():
    warp = bump((100, 100), 30.0, (4.0, 2.0))
    before = grid_points()

    left, started = left_by_repair(warp.forward(before), before, warp, before, (SIZE, SIZE))

    assert np.median(started) > 0.1
    assert np.median(left) < 0.1 * np.median(started)


def test_a_repair_through_points_that_saw_nothing_leaves_the_drift():
    warp = bump((100, 100), 30.0, (4.0, 2.0))
    before = grid_points()

    left, started = left_by_repair(before.copy(), before, warp, before, (SIZE, SIZE))

    assert np.median(left) == pytest.approx(np.median(started), rel=0.01)


def test_the_check_covers_the_whole_mask_not_only_the_kept_points():
    mask = np.zeros((SIZE, SIZE), bool)
    mask[50:150, 60:140] = True

    points = face_points(mask, step=10)

    assert len(points) == 80
    assert mask[points[:, 1].astype(int), points[:, 0].astype(int)].all()


def test_pixel_gap_reads_only_inside_the_mask():
    first = np.zeros((SIZE, SIZE, 3), np.uint8)
    second = first.copy()
    second[:50] = 200
    mask = np.zeros((SIZE, SIZE), bool)
    mask[100:] = True
    assert pixel_gap(first, second, mask) == 0.0
    mask[:50] = True
    assert pixel_gap(first, second, mask) > 0.0


def test_a_restyle_can_go_to_the_models():
    """The model processors refuse an array running backwards in memory."""
    assert all(stride > 0 for stride in restyle(dot_image(100, 100)).strides)


def test_a_shift_of_the_whole_face_is_not_left_as_drift():
    """Drift has the move, scale and turn taken out; what a repair left is
    counted the same way, so measuring every point 3 px off leaves none."""
    warp = bump((100, 100), 30.0, (4.0, 2.0))
    before = grid_points()
    measured_off = warp.forward(before) + [3.0, 0.0]

    left, started = left_by_repair(measured_off, before, warp, before, (SIZE, SIZE))

    assert np.median(left) < 0.1 * np.median(started)


def textured(size=SIZE, seed=3):
    noise = np.random.default_rng(seed).integers(0, 255, (size // 6, size // 6), np.uint8)
    grey = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    return np.dstack([grey] * 3)


def test_tracking_without_the_flow_finds_nothing_where_nothing_moved():
    image = textured()
    assert np.median(tracked_drift(image, image.copy(), grid_points())) == pytest.approx(0.0)


def test_tracking_without_the_flow_reads_a_sub_pixel_warp():
    warp = bump((100, 100), 25.0, (0.6, 0.0))
    before = grid_points()
    image = textured()

    read = np.median(tracked_drift(image, warp.render(image), before))
    true = np.median(score(warp.forward(before), before, warp).true["similarity"])

    assert read == pytest.approx(true, rel=0.25)
