"""A face mask supplied from outside, in place of the one `models.FaceMasker`
finds.

Any MASK can be wired into Measure: SameFace Face Mask's, with a chosen
person; a face parser's; a hand-painted one; a detector's box. Whatever is
inside it is measured as face - there are no ears to tell apart, so the
`ears` switch has nothing to act on.
"""
import cv2
import numpy as np

from .mask_shapes import _filled, _largest, outline
from .measure import Region

# A mask filling more of its own bounding box than this is a box, not a face:
# a bbox detector's, with hair and background inside it measured as face.
BOX_FILL = 0.95


def region_from_mask(mask):
    """A `Region` from a (h, w) mask: cut at one half, the largest piece kept
    and its holes filled - a parser's mask leaves the eyes and mouth out."""
    binary = (np.asarray(mask, np.float32) >= 0.5).astype(np.uint8)
    if not binary.any():
        raise ValueError("the mask is empty: nothing in it to measure")
    pieces = cv2.connectedComponents(binary, connectivity=8)[0] - 1
    region = _filled(_largest(binary))
    ys, xs = np.nonzero(region)
    box_area = (np.ptp(xs) + 1) * (np.ptp(ys) + 1)
    face = outline(region)
    return Region(mask=region, outline=face, parts={"face": face.tolist(), "ears": []},
                  supplied=True, pieces_dropped=pieces - 1,
                  box_shaped=bool(region.sum() / box_area > BOX_FILL))


def masks_for(masks, sizes, side):
    """One (h, w) mask or None per image of `sizes` [(h, w), ...].

    `masks` is None (not wired), one mask for every image, or one per image.
    """
    if masks is None:
        return [None] * len(sizes)
    masks = np.asarray(masks, np.float32)
    if masks.ndim == 2:
        masks = masks[None]
    if len(masks) not in (1, len(sizes)):
        raise ValueError(f"{len(masks)} {side} masks for {len(sizes)} images: "
                         "give one mask, or one per image")
    paired = [masks[0]] * len(sizes) if len(masks) == 1 else list(masks)
    for mask, (height, width) in zip(paired, sizes):
        if mask.shape != (height, width):
            raise ValueError(f"the {side} mask is {mask.shape[1]}x{mask.shape[0]} but its "
                             f"image is {width}x{height}: a mask must be its image's size")
    return paired
