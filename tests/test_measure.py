"""The pipeline end to end, with a mask and a flow made by hand.

No GPU and no models: `measure` takes the masker and the flow as arguments,
so a pair whose true drift is known can be put through every step that is
not a model - the crops, the round-trip check, the choice of points, the
frame, the fits and the repair.
"""
import cv2
import numpy as np
import pytest

from sameface.mask_shapes import join, outline, parts
from sameface.measure import (WORKING_SIZE, Region, chosen, measure, repair,
                              repair_frame)


def textured(width, height, seed=0):
    noise = np.random.default_rng(seed).integers(0, 255, (height // 8, width // 8, 3), np.uint8)
    return cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)


def oval_masker(rgb):
    """A face in the middle of every image, a third of its width across."""
    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (width // 6, height // 4), 0, 0, 360, 1, -1)
    region, _ = join(mask, [])
    return Region(mask=region, outline=outline(region), parts=parts(mask, [], region),
                  face_score=0.9, faces_found=1)


def still(first, second):
    zero = np.zeros((2, WORKING_SIZE, WORKING_SIZE), np.float32)
    return zero, zero.copy()


def a_patch_moved(dx):
    """Everything still except a square in the middle of the face, which the
    flow says moved `dx` working pixels to the right."""
    def flows(first, second):
        forward = np.zeros((2, WORKING_SIZE, WORKING_SIZE), np.float32)
        centre = WORKING_SIZE // 2
        forward[0, centre - 48:centre + 48, centre - 48:centre + 48] = dx
        backward = np.zeros_like(forward)
        backward[0, centre - 48:centre + 48, centre - 48 + int(dx):centre + 48 + int(dx)] = -dx
        return forward, backward
    return flows


def test_the_same_image_twice_has_no_drift():
    before = textured(320, 400)

    drift = measure(before, before.copy(), oval_masker, still)

    for fit in drift.fits.values():
        assert fit.highest == pytest.approx(0.0, abs=1e-6)
        assert fit.above_noise == 0.0
    assert drift.fits["similarity"].transform["scale"] == pytest.approx(1.0)


def test_a_resize_is_not_drift():
    """The after is the before at twice the size. Each side is cropped to its
    own face, so the flow sees nothing; brought into the before's frame, the
    points coincide."""
    before = textured(320, 400)
    after = cv2.resize(before, (640, 800), interpolation=cv2.INTER_CUBIC)

    drift = measure(before, after, oval_masker, still)

    assert drift.fits["similarity"].highest == pytest.approx(0.0, abs=1e-6)
    assert drift.fits["similarity"].transform["scale"] == pytest.approx(1.0, abs=1e-6)


def test_a_part_that_moved_is_drift_and_the_rest_is_not():
    before = textured(320, 400)

    drift = measure(before, before.copy(), oval_masker, a_patch_moved(6.0))

    fit = drift.fits["similarity"]
    assert fit.highest > 1.0
    # Most of the face did not move; the fit is not dragged far by the patch.
    assert fit.median < fit.highest / 3


def test_points_off_the_face_are_not_used():
    before = textured(320, 400)
    drift = measure(before, before.copy(), oval_masker, still)

    _, before_px, _ = chosen(drift.record)

    region = drift.before_region.mask
    inside = [region[int(round(y)), int(round(x))] for x, y in before_px]
    assert all(inside)
    assert len(before_px) < len(drift.record.points)


def test_a_repair_of_a_still_pair_is_the_after_again():
    before = textured(320, 400)
    after = before.copy()
    drift = measure(before, after, oval_masker, still)

    for framing in ("after", "before"):
        repaired, covered = repair(drift, framing=framing)

        assert repaired.shape[:2] == before.shape[:2]
        assert covered[10:-10, 10:-10].all()
        difference = np.abs(repaired[10:-10, 10:-10].astype(int) - after[10:-10, 10:-10])
        assert difference.mean() < 1.0


def test_a_repair_in_the_afters_frame_is_the_afters_size_and_all_covered():
    before = textured(320, 400)
    after = cv2.resize(before, (480, 600), interpolation=cv2.INTER_CUBIC)
    drift = measure(before, after, oval_masker, a_patch_moved(6.0))

    repaired, covered = repair(drift, framing="after")

    assert repaired.shape == after.shape
    assert covered.all()


def test_a_resize_repaired_in_the_afters_frame_is_the_after_again():
    """The before's points are carried into the after by the after's own
    pose. Carried the wrong way, the face would be moved and scaled."""
    before = textured(320, 400)
    after = cv2.resize(before, (640, 800), interpolation=cv2.INTER_CUBIC)
    drift = measure(before, after, oval_masker, still)

    repaired, _ = repair(drift, framing="after")

    difference = np.abs(repaired[10:-10, 10:-10].astype(int) - after[10:-10, 10:-10])
    assert difference.mean() < 1.0


def test_a_repair_in_the_afters_frame_moves_the_face_and_leaves_the_rest():
    before = textured(320, 400)
    after = before.copy()
    drift = measure(before, after, oval_masker, a_patch_moved(6.0))

    repaired, _ = repair(drift, framing="after")

    centre = (slice(170, 230), slice(130, 190))
    assert np.abs(repaired[centre].astype(int) - after[centre]).mean() > 5.0
    corner = (slice(0, 40), slice(0, 40))
    assert np.array_equal(repaired[corner], after[corner])


def test_a_repair_keeps_the_afters_detail():
    """An after twice the before's size is repaired at twice the size, not
    shrunk to the before's."""
    before = textured(320, 400)
    after = cv2.resize(before, (640, 800), interpolation=cv2.INTER_CUBIC)
    drift = measure(before, after, oval_masker, still)

    assert repair_frame(drift) == pytest.approx((640.0, 800.0), abs=1.0)


def test_the_map_and_the_report_are_drawn():
    from sameface import drawing
    before = textured(320, 400)
    drift = measure(before, before.copy(), oval_masker, a_patch_moved(6.0))

    for mode in ("similarity", "affine"):
        picture = drawing.shift_map(drift, mode, 0.05)
        assert picture.shape == before.shape and picture.dtype == np.uint8
    text = drawing.report(drift, 0.05)
    assert "similarity" in text and "affine" in text


def test_the_two_views_share_one_frame():
    from sameface.views import views
    before = textured(320, 400)
    drift = measure(before, before.copy(), oval_masker, a_patch_moved(6.0))
    frame = repair_frame(drift)

    for fit in ("similarity", "affine"):
        first, second = views(drift, frame, fit=fit, exaggeration=5)
        assert first.shape == second.shape == (int(frame[1]), int(frame[0]), 3)
        assert first.dtype == second.dtype == np.uint8


def test_points_that_did_not_come_back_are_counted_and_marked():
    """The patch's flow forward is not undone backward, so its points fail
    the round trip: left out of the fit, counted in the report, crossed on
    the before."""
    from sameface import drawing
    from sameface.measure import dropped
    from sameface.views import views

    def one_way(first, second):
        forward, _ = a_patch_moved(6.0)(first, second)
        return forward, np.zeros_like(forward)

    before = textured(320, 400)
    drift = measure(before, before.copy(), oval_masker, one_way)
    gaps = dropped(drift.record)

    assert len(gaps) > 0
    assert f"{len(gaps)} more dropped" in drawing.report(drift, 0.05)
    frame = repair_frame(drift)
    marked, _ = views(drift, frame, points=False, mesh=False, dropped=True)
    plain, _ = views(drift, frame, points=False, mesh=False, dropped=False)
    assert (marked != plain).any()
