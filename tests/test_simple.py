"""SameFace Simple: the advanced steps at their defaults, in one.

The same hand-made masks and flows as the pipeline's tests, and DIS optical
flow on the CPU where a real flow has to see the repaired pixels.
"""
import cv2
import numpy as np
from sameface.measure import measure
from sameface.plots import repair_histogram
from sameface.simple import report, simple, smaller_fit
from test_measure import a_patch_moved, dis_flow, oval_masker, textured


def bulged(image, pixels):
    """`image` with the middle of the face pushed `pixels` to the right and
    eased to nothing at its edge: drift a repair can take back out."""
    height, width = image.shape[:2]
    ys, xs = np.mgrid[:height, :width].astype(np.float32)
    distance = np.hypot((xs - width / 2) / (width / 6), (ys - height / 2) / (height / 4))
    push = pixels * np.clip(1 - distance, 0, 1) ** 2
    return cv2.remap(image, xs - push, ys, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def test_the_figures_are_measures_own():
    before, after = textured(320, 400), textured(320, 400, 1)
    flows = a_patch_moved(6)
    result = simple(before, after, oval_masker, flows, fix=False)
    alone = measure(before, after, oval_masker, flows)
    for mode in ("similarity", "affine"):
        assert result.drift.fits[mode].median == alone.fits[mode].median
    assert result.repaired is None and result.repaired_median is None


def test_the_fit_is_the_one_that_leaves_less():
    result = simple(textured(320, 400), textured(320, 400, 1), oval_masker, a_patch_moved(6),
                    fix=False)
    medians = {mode: fit.median for mode, fit in result.drift.fits.items()}
    assert result.fit == min(medians, key=medians.get) == smaller_fit(result.drift)
    assert result.median == medians[result.fit]


def test_a_repair_is_measured_again_and_drifts_less():
    before = textured(320, 400)
    after = bulged(before, 6)
    result = simple(before, after, oval_masker, dis_flow)
    assert result.repaired.shape == after.shape
    assert result.repaired_drift.after_size == result.drift.after_size
    assert result.repaired_median < result.median / 2


def test_the_report_gives_the_medians_and_nothing_else():
    before = textured(320, 400)
    result = simple(before, bulged(before, 6), oval_masker, dis_flow)
    lines = report(result).splitlines()
    assert lines == [f"SameFace drift: {result.median:.3f}% of face size",
                     f"after repair: {result.repaired_median:.3f}%"]


def test_the_report_warns_when_several_faces_were_found():
    def crowd(rgb):
        region = oval_masker(rgb)
        region.faces_found = 3
        return region
    result = simple(textured(320, 400), textured(320, 400, 1), crowd, a_patch_moved(6),
                    fix=False)
    text = report(result, "pair 1 of 2")
    assert text.startswith("SameFace drift - pair 1 of 2:")
    assert "faces found: 3 in the before and 3 in the after" in text
    assert len(text.splitlines()) == 2


def test_the_histogram_draws_the_edit_alone_or_against_the_repair():
    before = textured(320, 400)
    result = simple(before, bulged(before, 6), oval_masker, dis_flow)
    alone = repair_histogram(result.drift, result.fit, 0.01)
    both = repair_histogram(result.drift, result.fit, 0.01, result.repaired_drift)
    assert alone.shape == both.shape == (600, 1200, 3)
    assert not np.array_equal(alone, both)



def test_the_repair_is_measured_against_the_same_before_face():
    before = textured(320, 400)
    result = simple(before, bulged(before, 6), oval_masker, dis_flow)
    assert result.repaired_drift.before_region is result.drift.before_region
