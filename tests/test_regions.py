"""A face mask supplied from outside, in place of the one SameFace finds.

Any node's MASK can be wired in: a face parser's, a hand-painted one, a
bounding box. Whatever is inside it is measured as face. The masks are drawn
here by hand, so each test knows its answer.
"""
import cv2
import numpy as np
import pytest

from sameface.mask_shapes import join, outline, parts
from sameface.measure import Region
from sameface.regions import masks_for, region_from_mask

SHAPE = (400, 300)


def ellipse(cx=150, cy=200, ax=60, ay=90, shape=SHAPE):
    mask = np.zeros(shape, np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1, -1)
    return mask


def test_a_mask_becomes_a_region_that_is_all_face():
    region = region_from_mask(ellipse().astype(np.float32))

    assert isinstance(region, Region)
    assert region.supplied
    assert np.array_equal(region.mask, ellipse())
    assert region.parts["ears"] == []
    assert region.parts["face"] == outline(region.mask).tolist()


def test_a_soft_mask_is_cut_at_one_half():
    soft = ellipse().astype(np.float32) * 0.6 + ellipse(ax=80, ay=110).astype(np.float32) * 0.3

    assert np.array_equal(region_from_mask(soft).mask, ellipse())


def test_of_two_pieces_the_larger_is_kept_and_the_other_counted():
    two = ellipse() | ellipse(cx=270, cy=40, ax=20, ay=20)

    region = region_from_mask(two)

    assert np.array_equal(region.mask, ellipse())
    assert region.pieces_dropped == 1


def test_holes_are_filled():
    """A face parser leaves the eyes and mouth out; the face is still one piece."""
    holed = ellipse()
    cv2.circle(holed, (150, 180), 15, 0, -1)

    assert np.array_equal(region_from_mask(holed).mask, ellipse())


def test_a_box_is_flagged_and_an_oval_is_not():
    """A bbox detector's mask is a rectangle: hair and background inside it
    are measured as face, and the report has to say so."""
    square = np.zeros(SHAPE, np.uint8)
    square[100:300, 80:220] = 1

    assert region_from_mask(square).box_shaped
    assert not region_from_mask(ellipse()).box_shaped


def test_an_empty_mask_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        region_from_mask(np.zeros(SHAPE, np.float32))


def test_a_supplied_mask_gives_the_region_our_own_masker_would():
    """The same pixels, whichever way they arrive."""
    mask = ellipse()
    joined, _ = join(mask, [])
    ours = Region(mask=joined, outline=outline(joined), parts=parts(mask, [], joined))

    theirs = region_from_mask(mask)

    assert np.array_equal(theirs.mask, ours.mask)
    assert np.array_equal(theirs.outline, ours.outline)
    assert theirs.parts == ours.parts


# ---------- a mask batch against an image batch ----------


def test_no_mask_is_none_for_every_image():
    assert masks_for(None, [SHAPE, SHAPE], "before") == [None, None]


def test_one_mask_serves_every_image():
    mask = ellipse()[None]

    found = masks_for(mask, [SHAPE] * 3, "after")

    assert len(found) == 3 and all(np.array_equal(each, ellipse()) for each in found)


def test_a_mask_per_image_is_paired_in_order():
    masks = np.stack([ellipse(), ellipse(cx=160)])

    found = masks_for(masks, [SHAPE, SHAPE], "after")

    assert np.array_equal(found[1], ellipse(cx=160))


def test_a_single_mask_without_a_batch_dimension_is_one_mask():
    assert len(masks_for(ellipse(), [SHAPE, SHAPE], "before")) == 2


def test_a_mask_batch_that_fits_neither_one_nor_the_images_is_refused():
    with pytest.raises(ValueError, match="2 before masks for 3 images"):
        masks_for(np.stack([ellipse(), ellipse()]), [SHAPE] * 3, "before")


def test_a_mask_of_the_wrong_size_is_refused_naming_both_sizes():
    with pytest.raises(ValueError, match=r"after mask is 300x400.*image is 600x800"):
        masks_for(ellipse()[None], [(800, 600)], "after")
