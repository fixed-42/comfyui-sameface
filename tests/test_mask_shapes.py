"""Putting a face mask together from a face and its ears.

A text-prompted detector finds "face" and "ear" boxes and a segmenter turns
each into a mask. What is left is plain geometry, and it is where the choices
are: which ear boxes belong to this face, and what to do with an ear whose
mask does not touch the face's. No models here - the masks are drawn by hand,
so each test knows its answer.
"""
import numpy as np
import pytest

from sameface.mask_shapes import ears_beside, face_in, join, outline, parts, pick_face

FACE = [100.0, 100.0, 300.0, 400.0]      # 200 wide, 300 tall


def box(cx, cy, w=40, h=70):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ---------- which face, when there are several ----------

# A family photo as the detector saw it: a poster on the wall, then a baby,
# the mother and the father, left to right. The father is barely the surest.
POSTER, BABY = (box(110, 75, 50, 43), 0.43), (box(245, 408, 103, 135), 0.59)
MOTHER, FATHER = (box(431, 359, 118, 160), 0.63), (box(698, 300, 131, 193), 0.65)
FAMILY = [MOTHER, POSTER, FATHER, BABY]


def test_the_surest_face_comes_first():
    assert pick_face(FAMILY, "surest", 0) == FATHER
    assert pick_face(FAMILY, "surest", 1) == MOTHER


def test_left_to_right_counts_from_the_left_edge():
    """By the middle of each box, so a tall face starting further left than
    a small one beside it does not jump the queue."""
    assert [pick_face(FAMILY, "left to right", index) for index in range(4)] == \
        [POSTER, BABY, MOTHER, FATHER]


def test_a_face_past_the_last_is_an_error_that_says_how_many_there_are():
    with pytest.raises(ValueError, match="4 faces found"):
        pick_face(FAMILY, "left to right", 4)


def test_heads_are_counted_the_same_way_and_named_in_the_error():
    assert pick_face(FAMILY, "left to right", 2, "head") == MOTHER
    with pytest.raises(ValueError, match="head 4 asked for, but 4 heads found"):
        pick_face(FAMILY, "left to right", 4, "head")


def test_a_heads_face_is_the_one_inside_it():
    """SAM 2 asked for a head outlines the hair; the face is joined from the
    face box inside the head box - not a neighbour's."""
    mothers_head = box(431, 330, 200, 210)

    assert face_in(mothers_head, FAMILY) == MOTHER


def test_a_head_with_no_face_inside_has_none():
    assert face_in(box(431, 700, 100, 60), FAMILY) is None


def test_no_face_is_an_error():
    with pytest.raises(ValueError, match="no face found"):
        pick_face([], "surest", 0)


def test_an_unknown_order_is_an_error():
    with pytest.raises(ValueError, match="order"):
        pick_face(FAMILY, "tallest", 0)


# ---------- which ears belong to the face ----------


def test_an_ear_beside_the_face_is_its_ear():
    left, right = (box(90, 250), 0.6), (box(310, 250), 0.5)

    assert ears_beside(FACE, [left, right]) == [left, right]


def test_an_ear_far_from_the_face_is_someone_elses_or_not_an_ear():
    """The detector is asked for "ear" over the whole image. A second person,
    or a curl of hair it mistook, must not be glued onto this face."""
    assert ears_beside(FACE, [(box(600, 250), 0.9)]) == []
    assert ears_beside(FACE, [(box(200, 600), 0.9)]) == []     # below the chin


def test_a_box_as_big_as_the_face_is_not_an_ear():
    assert ears_beside(FACE, [(box(200, 250, w=180, h=280), 0.9)]) == []


def test_at_most_two_ears_and_the_surest_two():
    found = [(box(90, 250), 0.3), (box(310, 250), 0.8), (box(95, 200), 0.6)]

    assert [score for _, score in ears_beside(FACE, found)] == [0.8, 0.6]


# ---------- joining them ----------


def disc(shape, cx, cy, r):
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    return ((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r).astype(np.uint8)


SHAPE = (400, 400)


def test_an_ear_touching_the_face_becomes_part_of_one_region():
    face, ear = disc(SHAPE, 200, 200, 100), disc(SHAPE, 310, 200, 25)

    mask, kept = join(face, [ear])

    assert kept == [True]
    assert mask[200, 325] == 1 and mask[200, 200] == 1


def test_an_ear_a_few_pixels_short_of_the_face_is_still_joined():
    """Two masks cut separately rarely meet exactly. Left as they are, keeping
    the largest piece would throw the ear away over a hairline gap."""
    face, ear = disc(SHAPE, 200, 200, 100), disc(SHAPE, 333, 200, 25)   # 8px gap
    assert not (face & ear).any()

    mask, kept = join(face, [ear])

    assert kept == [True] and mask[200, 340] == 1


def test_an_ear_well_clear_of_the_face_is_left_out_and_reported():
    """On a profile the ear sits behind the jaw with hair between. Bridging
    that far would annex whatever lies between, so the ear is dropped - and
    said to be dropped, rather than the mask quietly being ears-out."""
    face, ear = disc(SHAPE, 150, 200, 80), disc(SHAPE, 350, 200, 20)     # 100px gap

    mask, kept = join(face, [ear])

    assert kept == [False]
    assert mask[200, 350] == 0 and mask[200, 150] == 1


def test_the_result_is_one_region_with_no_holes():
    face = disc(SHAPE, 200, 200, 100)
    face[190:210, 190:210] = 0                      # a hole, as a nostril might leave

    mask, _ = join(face, [])

    assert mask[200, 200] == 1


def test_no_ears_is_just_the_face():
    face = disc(SHAPE, 200, 200, 100)

    mask, kept = join(face, [])

    assert kept == [] and (mask == face).all()


# ---------- the outline handed to RAFT ----------


def test_the_outline_is_a_polygon_that_keeps_concavities():
    """It is given to RAFT as a boundary, as is. Hulling it would fill in the
    notch between a face and its ear."""
    face, ear = disc(SHAPE, 200, 200, 100), disc(SHAPE, 310, 200, 25)
    mask, _ = join(face, [ear])

    points = outline(mask)

    import cv2
    hull = cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
    assert points.shape[1] == 2 and 8 < len(points) < 200
    assert cv2.contourArea(points.astype(np.float32)) < hull * 0.995


def test_an_empty_mask_has_no_outline_and_says_so():
    with pytest.raises(ValueError, match="no face"):
        outline(np.zeros(SHAPE, np.uint8))


# ---------- what the region was made of ----------
#
# One outline is what RAFT is cropped to and what is drawn. Whether the ears
# feed a fit is chosen afterwards, so the provider also says which part of
# that outline is face and which is ear.


def test_the_parts_are_the_face_alone_and_each_ear():
    face, ear = disc(SHAPE, 200, 200, 100), disc(SHAPE, 310, 200, 25)
    region, _ = join(face, [ear])

    found = parts(face, [ear], region)

    face_xs = np.asarray(found["face"])[:, 0]
    ear_xs = np.asarray(found["ears"][0]["points"])[:, 0]
    assert face_xs.max() <= 301                      # stops where the face does
    assert ear_xs.min() >= 284 and ear_xs.max() >= 334
    assert found["ears"][0]["joined"] is True


def test_an_ear_left_out_of_the_region_is_still_handed_over():
    """`join` leaves a distant ear out of the one outline. It was found all
    the same, and saying where is what lets it be chosen later."""
    face, near, far = (disc(SHAPE, 150, 200, 80), disc(SHAPE, 240, 200, 20),
                       disc(SHAPE, 350, 200, 20))
    region, _ = join(face, [near, far])

    found = parts(face, [near, far], region)

    assert [ear["joined"] for ear in found["ears"]] == [True, False]
    assert np.asarray(found["ears"][1]["points"])[:, 0].min() >= 329


def test_with_no_ears_the_face_part_is_the_whole_region():
    face = disc(SHAPE, 200, 200, 100)
    mask, _ = join(face, [])

    found = parts(face, [], mask)

    assert found["ears"] == []
    assert found["face"] == outline(mask).tolist()


def test_an_ear_box_the_segmenter_found_nothing_in_has_no_outline():
    face, nothing = disc(SHAPE, 200, 200, 100), np.zeros(SHAPE, np.uint8)
    region, _ = join(face, [nothing])

    assert parts(face, [nothing], region)["ears"] == []


def test_an_ear_is_joined_only_if_the_piece_outlined_is():
    """A segmenter's mask can carry a speck away from its main piece. If the
    speck touches the face and the ear proper does not, the ear is apart."""
    face = disc(SHAPE, 150, 200, 80)
    ear = disc(SHAPE, 350, 200, 20) | disc(SHAPE, 232, 200, 3)
    region, kept = join(face, [ear])
    assert kept == [True]                            # the speck touched

    found = parts(face, [ear], region)

    assert found["ears"][0]["joined"] is False
    assert np.asarray(found["ears"][0]["points"])[:, 0].min() >= 329
