"""Putting a face mask together from a face and its ears.

The detecting and the segmenting are models and live in `models.py`. What is
left is geometry, kept here with no torch in it so it can be tested without a
GPU - and because this is where the choices are that decide the mask: which
"ear" boxes belong to this face, and what becomes of an ear whose mask does
not reach the face's.
"""
import cv2
import numpy as np

# An ear sits beside the face: its centre within the face box widened by this
# share of the face's width on each side, and between its top and bottom.
BESIDE = 0.4
# ...and is far smaller than the face. A box near the size of the face is the
# detector answering "ear" with the head.
EAR_WIDTH, EAR_HEIGHT = 0.45, 0.6
MAX_EARS = 2

# How the faces in a picture are counted, for choosing one of several people.
# "surest" is the detector's own confidence; "left to right" goes by the
# middle of each box, and is the one that stays put when two scores are close.
ORDERS = ("surest", "left to right")
# What is masked: the face (and its ears), or the whole head, hair included -
# for an edit that left the hair alone, so the head's shape is measured too.
REGIONS = ("face", "head")

# How far apart an ear and the face may be and still be joined, as a share of
# the face mask's diagonal. Two masks cut separately seldom meet exactly;
# this bridges a hairline gap without reaching across a cheek.
JOIN_REACH = 0.04
# Douglas-Peucker tolerance as a share of the perimeter: tens of vertices
# rather than thousands, with the notches still in.
EPSILON_FRAC = 0.0015


def pick_face(faces, order="surest", index=0, what="face"):
    """The `(box, score)` at `index` among `faces`, counted in `order`.
    `what` names them in an error: a face, or a head."""
    if order == "surest":
        ranked = sorted(faces, key=lambda face: -face[1])
    elif order == "left to right":
        ranked = sorted(faces, key=lambda face: face[0][0] + face[0][2])
    else:
        raise ValueError(f"unknown face order {order!r}: one of {', '.join(ORDERS)}")
    if not ranked:
        raise ValueError(f"no {what} found in the image")
    if not 0 <= index < len(ranked):
        raise ValueError(f"{what} {index} asked for, but {len(ranked)} {what}s found "
                         f"(0 to {len(ranked) - 1}, {order})")
    return ranked[index]


def face_in(head_box, faces):
    """The `(box, score)` among `faces` most inside `head_box`, or None.

    Asked for a head, SAM 2 outlines the hair and leaves the face out, so a
    head mask is the hair joined to this face's own mask.
    """
    def inside(face):
        x0, y0, x1, y1 = face[0]
        width = max(0.0, min(x1, head_box[2]) - max(x0, head_box[0]))
        height = max(0.0, min(y1, head_box[3]) - max(y0, head_box[1]))
        return width * height / max((x1 - x0) * (y1 - y0), 1e-9)
    best = max(faces, key=inside, default=None)
    return best if best is not None and inside(best) > 0.5 else None


def ears_beside(face_box, found):
    """The detected ears that belong to this face, surest first.

    `found` is `(box, score)` pairs from asking for "ear" over the whole
    image, which also returns other people's ears and the odd curl of hair.
    """
    x0, y0, x1, y1 = face_box
    width, height = x1 - x0, y1 - y0
    beside = []
    for box, score in found:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if (x0 - BESIDE * width <= cx <= x1 + BESIDE * width and y0 <= cy <= y1
                and box[2] - box[0] < EAR_WIDTH * width
                and box[3] - box[1] < EAR_HEIGHT * height):
            beside.append((box, score))
    return sorted(beside, key=lambda ear: -ear[1])[:MAX_EARS]


def _ellipse(size):
    size = max(3, int(size) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def _filled(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
    return filled


def join(face, ears, reach=JOIN_REACH):
    """One region from a face mask and its ear masks, and which ears made it.

    An ear is kept when it ends up connected to the face after a small gap is
    bridged. One that stays apart - a profile's ear, behind the jaw - is left
    out and reported as such: the region handed to RAFT is a single outline,
    and reaching across to a distant ear would take in whatever lies between.
    """
    face = _largest(np.asarray(face, np.uint8))
    if not ears:
        return _filled(face), []
    both = face.copy()
    for ear in ears:
        both |= np.asarray(ear, np.uint8)
    ys, xs = np.nonzero(face)
    diagonal = float(np.hypot(np.ptp(xs) + 1, np.ptp(ys) + 1)) if len(xs) else 0.0
    bridged = cv2.morphologyEx(both, cv2.MORPH_CLOSE, _ellipse(diagonal * reach))
    region = _largest(bridged)
    kept = [bool((region & np.asarray(ear, np.uint8)).any()) for ear in ears]
    return _filled(region), kept


def parts(face, ears, region):
    """What `region` was made of: the face alone, and each ear found.

    `join` answers with one outline, which is what a flow is cropped to and
    what is drawn. Whether the ears feed a fit is chosen after the measuring,
    so the pieces are handed over as well. An ear `join` left out is here all
    the same, marked as apart: it was found, it is only not in the one outline.

    `joined` is asked of the piece that is outlined, not of the ear mask as a
    whole: a stray speck of it touching the face does not put the ear in.
    """
    face_alone, _ = join(face, [])
    region = np.asarray(region, np.uint8)
    found = []
    for ear in ears:
        ear = np.asarray(ear, np.uint8)
        # A box the segmenter found nothing in has no outline to give.
        if ear.any():
            piece = _largest(ear)
            found.append({"points": outline(piece).tolist(),
                          "joined": bool((region & piece).any())})
    return {"face": outline(face_alone).tolist(), "ears": found}


def outline(mask, epsilon_frac=EPSILON_FRAC):
    """The region's outer boundary, simplified but still concave."""
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("no face region found in this image")
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
