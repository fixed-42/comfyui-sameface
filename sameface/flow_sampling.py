"""The part of a flow measurement that needs no model.

Where to look, how the two sides are brought into one frame, and which sampled
points are kept.

**No torch in this file, deliberately.** It is the part that can be wrong
rather than merely unavailable, so it is tested without a GPU.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Fraction of the face outline's extent added on each side. The face box is a
# region of interest, not a tight crop: flow at the boundary needs context.
BOX_PAD = 0.25


def face_box(points, width, height, pad=BOX_PAD):
    """A padded box around a set of points, clamped to the image.

    Returned as floats: the crop is resampled anyway, so rounding here would
    quantise the working frame for no gain and make the two sides of a pair
    disagree by up to a pixel before the flow even runs.
    """
    points = np.asarray(points, float)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0.0, x0 - dx), max(0.0, y0 - dy),
            min(float(width), x1 + dx), min(float(height), y1 + dy))


def crop_points(region, parts=None):
    """Every point the crop has to take in: the region, and any ear apart.

    The crop is the face and whatever ears were found, whichever of them go
    on to feed a fit. Leaving a distant ear out would make "ears in" a
    different crop from "ears out" - a different grid and a different run -
    where the choice is meant to be made afterwards, over one measurement.

    A joined ear is inside the region already and adds nothing here. Adding
    its own outline anyway could move the box by the pixel or so that two
    simplified outlines of the same edge differ by.
    """
    points = [np.asarray(region, float)]
    for ear in (parts or {}).get("ears", []):
        if not ear["joined"]:
            points.append(np.asarray(ear["points"], float))
    return np.vstack(points)


def crop_to_working(image, box, size):
    """Resample the box out of `image` into a `size` x `size` frame.

    Returns the frame and the affine that maps working-frame coordinates back
    to image pixels. **That inverse is the whole reason this is legitimate.**
    Coarse-aligning the two sides is what lets RAFT see motion in its trained
    range; discarding the transform afterwards would also discard the global
    similarity being measured, and the null test would come back green having
    measured nothing.
    """
    scale_x, scale_y, x0, y0 = transform = box_transform(box, size)
    matrix = np.array([[1.0 / scale_x, 0.0, -x0 / scale_x],
                       [0.0, 1.0 / scale_y, -y0 / scale_y]], dtype=np.float32)
    warped = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_AREA
                            if scale_x > 1 else cv2.INTER_LINEAR)
    return warped, transform


def box_transform(box, size):
    """How working-frame pixels map back to the image: scale, then offset.

    On its own so that a run's recorded box is enough to place its points in
    the image again, with no image to hand - by the arithmetic that cut the
    crop, not a copy of it.
    """
    x0, y0, x1, y1 = box
    return ((x1 - x0) / size, (y1 - y0) / size, x0, y0)


def to_image_pixels(points, transform):
    scale_x, scale_y, x0, y0 = transform
    return np.column_stack([points[:, 0] * scale_x + x0,
                            points[:, 1] * scale_y + y0])


def to_working_pixels(points, transform):
    """The inverse of `to_image_pixels`, for taking a region *into* the frame.

    The grid is built in working-frame coordinates, so the hull has to meet it
    there rather than the other way around - mapping 961 points out to compare
    them against the hull would do the same arithmetic hundreds of times over.
    """
    scale_x, scale_y, x0, y0 = transform
    points = np.asarray(points, float)
    return np.column_stack([(points[:, 0] - x0) / scale_x,
                            (points[:, 1] - y0) / scale_y])


def region_contour(points, transform, convex):
    """The region as a working-frame contour, ready for a point test.

    Unpadded, deliberately: `BOX_PAD` exists so the *crop* carries context for
    flow at the boundary, which is a different question from which samples
    count. Padding the region too would put back a share of exactly what this
    is removing.

    `convex` is the whole difference between `hull` and `polygon`. Points from
    a landmark provider are an unordered cloud and have to be hulled to become
    a boundary at all; points from a region provider already *are* a boundary,
    in order, and hulling them would discard the concavities that make a face
    outline a face rather than a blob.
    """
    points = np.asarray(points, np.float32)
    if convex:
        points = cv2.convexHull(points).reshape(-1, 2)
    return to_working_pixels(
        points, transform).astype(np.float32).reshape(-1, 1, 2)


@dataclass(frozen=True)
class Sampled:
    """Every grid point that was asked about, and what became of it.

    All of them, not only the kept ones: which points were dropped, and why,
    is the first thing to look at when a pair keeps few.
    """

    origin: np.ndarray     # (n, 2) where each point started, working pixels
    landed: np.ndarray     # (n, 2) where the forward flow put it
    error: np.ndarray      # (n,) how far the round trip missed its start
    keep: np.ndarray       # (n,) bool: the round trip came back within limit
    outside: np.ndarray    # (n,) bool: it landed off the working frame


def sample(flow_ab, flow_ba, grid_step, consistency_px):
    """Every grid point in the crop, its displacement, and whether it is
    self-consistent.

    The whole crop, not the part of it inside the region: which points feed a
    fit is chosen afterwards (see `tags`), so the flow is read once and that
    choice never needs another run.

    The consistency check is the confidence RAFT does not provide: follow the
    forward flow and come back along the reverse one, and see whether you
    arrive where you started.
    """
    size = flow_ab.shape[1]
    half = grid_step // 2
    coords = np.arange(half, size - half, grid_step)
    grid_x, grid_y = np.meshgrid(coords, coords)
    xs, ys = grid_x.ravel(), grid_y.ravel()

    dx, dy = flow_ab[0, ys, xs], flow_ab[1, ys, xs]
    forward = np.column_stack([xs + dx, ys + dy])

    # Where the forward flow lands, read the reverse flow back. Nearest
    # neighbour rather than bilinear: this is a rejection test, and half a
    # pixel of sampling error does not change whether a point is trustworthy.
    back_x = np.clip(np.rint(forward[:, 0]).astype(int), 0, size - 1)
    back_y = np.clip(np.rint(forward[:, 1]).astype(int), 0, size - 1)
    returned = np.column_stack([forward[:, 0] + flow_ba[0, back_y, back_x],
                                forward[:, 1] + flow_ba[1, back_y, back_x]])
    origin = np.column_stack([xs, ys])
    error = np.hypot(*(returned - origin).T)

    # **Recorded, and nothing more.** A point that lands off the frame has its
    # reverse flow read at the clamped edge pixel above, and can pass the
    # check by accident. That is how the reference figures were taken, so it
    # is left alone here and only labelled - whether such points should count
    # is a question the labels make cheap to ask, not one to settle by quietly
    # changing which points are kept.
    outside = ((forward < 0) | (forward > size - 1)).any(axis=1)
    return Sampled(origin=origin, landed=forward, error=error,
                   keep=error <= consistency_px, outside=outside)


def outcomes(sampled):
    """One word per sampled point: kept or dropped, and whether it left."""
    return [("kept" if keep else "dropped")
            + ("_outside" if outside else "" if keep else "_inconsistent")
            for keep, outside in zip(sampled.keep, sampled.outside)]


def _inside(points, contour):
    return np.array([cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
                     for x, y in points], bool)


def tags(origin, region=None, face=None, ears=()):
    """`face`, `ear` or `outside` for each grid point, in the before frame.

    `region` is the one outline a run is cropped to - face and ears joined.
    `face` is the face alone, where the region's provider said which part is
    which, and `ears` are the outlines of ears standing apart from the region.
    All three are working-frame contours, as `region_contour` makes them.

    Inside the region a point is `face` if the face alone covers it and `ear`
    otherwise: what the joining added is the ear's. That keeps "face or ear"
    exactly the set of points the region contains, which is what was sampled
    before the whole crop was. Outside the region only an ear apart is
    anything; the rest is `outside`.

    A point outside is not a measurement that failed. It is one the mask did
    not ask for, recorded so that a person can.
    """
    if region is None:
        return ["face"] * len(origin)
    in_region = _inside(origin, region)
    on_face = in_region if face is None else in_region & _inside(origin, face)
    on_ear = in_region & ~on_face
    for ear in ears:
        on_ear |= ~in_region & _inside(origin, ear)
    return ["face" if face_ else "ear" if ear_ else "outside"
            for face_, ear_ in zip(on_face, on_ear)]
