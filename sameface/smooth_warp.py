"""The smooth warp: moving an after image's features back, part by part.

A similarity or an affine moves the whole after image as one sheet. What it
leaves is local - a nose, an ear, a jaw sitting somewhere else relative to
the rest - and no sheet can undo that. This moves each part by its own
amount.

**How.** Every matched point says how far the image is off at its place. A
pixel moves by a smooth blend of what the points near it say: all the way at
a point, less with distance, not at all once the neighbouring points are
reached if those did not move. It is a thin-plate spline - a thin sheet
pinned at each point and left to bend as gently as it can between them. No
triangles, so none of the creases a triangle mesh leaves along its edges.

**It gives way.** The sheet is not forced through every point. One point
that moved against still neighbours is a wrong match far more often than a
feature, and following it exactly would dent the face; `SMOOTHING` lets the
sheet miss each point a little, which flattens a spike and leaves a movement
several points share alone.

**It stops.** The blend is eased to nothing just beyond the points, so what
lies outside the face is left exactly where the flat fit put it.

**It cannot grade itself.** A warp built from these points puts these points
where they belong by construction. Whether the repaired image is better is
answered by measuring it again against the before - a fresh run on new
pixels - and by looking.
"""
import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from .geometry import cover

# How readily the sheet gives way, with the points' spacing as the unit of
# length. Zero passes through every point. Chosen on hand-made cases: at 1.0
# a lone point moving against still neighbours is followed about half way,
# two neighbours moving together about 70%, and a patch of points moving
# together all of it.
SMOOTHING = 1.0
# How far beyond the points the blend takes to ease to nothing, in spacings.
FADE = 2.0


def spacing_of(points):
    """How far a point typically is from its nearest neighbour."""
    points = np.asarray(points, float)
    if len(points) < 2:
        return 0.0
    nearest, _ = cKDTree(points).query(points, k=2)
    return float(np.median(nearest[:, 1]))


def field(points, offsets, frame_size, smoothing=SMOOTHING, fade=FADE):
    """How far to move at every pixel of a frame: `offsets` at `points`,
    blended smoothly between them and eased to nothing beyond them.

    Returns `(height, width, 2)`, in the units the offsets are given in.
    """
    points, offsets = np.asarray(points, float), np.asarray(offsets, float)
    width, height = (int(round(side)) for side in frame_size)
    spacing = spacing_of(points)
    if spacing <= 0 or not np.any(offsets):
        return np.zeros((height, width, 2))

    # The sheet, solved with the spacing as the unit so `smoothing` means the
    # same thing whatever size the image is.
    sheet = RBFInterpolator(points / spacing, offsets, kernel="thin_plate_spline",
                            smoothing=smoothing, degree=1)
    # Evaluated on a coarse grid and enlarged: the sheet is smooth on the
    # scale of the points, so nothing is lost and a megapixel frame costs the
    # same as a small one.
    step = max(2.0, spacing / 4.0)
    xs = np.arange(0.0, width + step, step)
    ys = np.arange(0.0, height + step, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coarse = sheet(np.column_stack([grid_x.ravel(), grid_y.ravel()]) / spacing)
    coarse = coarse.reshape(len(ys), len(xs), 2).astype(np.float32)
    # A coarse sample sits at a pixel, not at the centre of a cell, so the
    # enlargement is an explicit resampling at each pixel's own position.
    pixel_x, pixel_y = np.meshgrid(np.arange(width, dtype=np.float32) / step,
                                   np.arange(height, dtype=np.float32) / step)
    blended = cv2.remap(coarse, pixel_x, pixel_y, cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)

    # Eased to nothing beyond the points. Within a spacing of a point the
    # blend is whole; from there it fades over `fade` spacings.
    away = np.ones((height, width), np.uint8)
    inside = ((points >= 0) & (points < [width, height])).all(axis=1)
    marks = np.round(points[inside]).astype(int)
    away[np.clip(marks[:, 1], 0, height - 1), np.clip(marks[:, 0], 0, width - 1)] = 0
    distance = cv2.distanceTransform(away, cv2.DIST_L2, 5)
    weight = np.clip(1.0 - (distance - spacing) / (fade * spacing), 0.0, 1.0)
    weight = weight * weight * (3.0 - 2.0 * weight)          # no corner where it starts
    return blended.astype(float) * weight[..., None]


class Warp:
    """The flat fit and the smooth warp on top of it, in a frame.

    `source` and `target` are the matched points in the frame - after and
    before - and `matrix` the flat fit that carries after onto before there.

    **Worked backwards, which is what gets the direction right.** For each
    pixel of the output the question is which after pixel belongs there. The
    flat fit alone says `inverse(matrix)` of it; a matched point says exactly,
    because the after point `source` is what belongs at `target`. The
    difference between the two answers, at each `target`, is what the sheet
    blends. Building the field forwards from the after side and then having
    to invert it is how a warp ends up pushing a feature the same distance
    the wrong way.
    """

    def __init__(self, source, target, matrix, frame_size, smoothing=SMOOTHING):
        source, target = np.asarray(source, float), np.asarray(target, float)
        self.frame_size = frame_size
        self.width, self.height = (int(round(side)) for side in frame_size)
        self.matrix = np.asarray(matrix, float)
        full = np.vstack([self.matrix, [0.0, 0.0, 1.0]])
        self.back = np.linalg.inv(full)[:2]
        self.extra = field(target, source - self.flat(target),
                           (self.width, self.height), smoothing=smoothing)

    def flat(self, points):
        """Frame points to where the flat fit alone takes them from, in after."""
        return points @ self.back[:, :2].T + self.back[:, 2]

    def render(self, image, strength=1.0):
        """`image` - the after - moved, with an alpha channel added: zero where
        no after pixel landed, which is the only way to tell "uncovered" from
        "black"."""
        grid_x, grid_y = np.meshgrid(np.arange(self.width, dtype=float),
                                     np.arange(self.height, dtype=float))
        wanted = self.flat(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        wanted = wanted.reshape(self.height, self.width, 2) + strength * self.extra
        # Frame coordinates back to the image's own pixels.
        scale, offset = cover(image.shape[1::-1], self.frame_size)
        in_image = ((wanted + offset) / scale).astype(np.float32)

        with_alpha = np.dstack([image, np.full(image.shape[:2], 255, image.dtype)])
        return cv2.remap(with_alpha, in_image[..., 0], in_image[..., 1], cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def moved(self, source, strength=1.0, rounds=20):
        """Where after points `source` (in the frame) end up in the render.

        The render says, for each output pixel, which after point lands there;
        this answers the other way round, by stepping: from where the flat fit
        alone would put a point, correct by the warp found there, until it
        settles. The sheet is smooth on the scale of the points, so a few
        rounds are plenty. What it gives is where the warp put a point, not a
        measurement of the repaired image.
        """
        source = np.asarray(source, float)
        linear, shift = self.matrix[:, :2], self.matrix[:, 2]
        landed = source @ linear.T + shift
        extra = self.extra.astype(np.float32)
        for _ in range(rounds):
            at = np.clip(landed, 0, [self.width - 1, self.height - 1]).astype(np.float32)
            push = cv2.remap(extra, at[:, 0:1], at[:, 1:2], cv2.INTER_LINEAR).reshape(-1, 2)
            landed = (source - strength * push) @ linear.T + shift
        return landed


def render_repaired(image, source, target, matrix, frame_size, strength=1.0,
                    smoothing=SMOOTHING):
    """The after image, moved by the flat fit and then by the smooth warp.

    See `Warp`. Returns the image with an alpha channel added.
    """
    return Warp(source, target, matrix, frame_size, smoothing).render(image, strength)
