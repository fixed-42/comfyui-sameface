"""The before and the after in one frame, with the measured points drawn on.

Two pictures the same size, for a slider: the before with its points, and
the after moved by the flat fit, with where its points ended up and an arrow
from each to where the before has it. To see a repair, measure the repaired
image against the before and view that: the points are then measured on the
repaired pixels, not placed by the warp that made them. A mesh joins the
points into triangles, the same triangles on both sides, so a part of the
face that drifted shows as triangles that change shape across the slider.

**The colours are the viewer's.** Green is before, red is after; grey is a
point within the noise floor - measured, found nothing.

**A gap in the grid is a point the flow could not vouch for.** Each point
is followed forward and back; one that does not come back to where it
started is left out of the measurement, and drawn on the before as an orange
cross. Not on the after: where it would go there is exactly the flow that
was not believed.

**Exaggeration stretches the arrows and nothing else.** The dots and the
mesh stay where the points are, so the mesh always sits on the dots. The
factor is printed on the picture, because an unlabelled exaggeration is a
lie.
"""
import cv2
import numpy as np
from scipy.spatial import Delaunay

from . import geometry
from .measure import NOISE_FLOOR, chosen, dropped as dropped_points
from .smooth_warp import Warp, spacing_of

BEFORE = (0x00, 0xe6, 0x76)
AFTER = (0xff, 0x3b, 0x30)
QUIET = (0x8c, 0x96, 0xa8)
DROPPED = (0xd9, 0x59, 0x26)
MESH_BEFORE = (0x9b, 0xff, 0xc8)
MESH_AFTER = (0xff, 0xa8, 0xa0)
EXAGGERATIONS = (1, 5, 10, 20)
# The marks are drawn this many times larger than the picture and scaled
# down onto it, which is what makes a line finer than one pixel possible.
SUPERSAMPLE = 4
# In picture pixels, at a 1000 px frame; they scale with the frame.
MESH_WIDTH = 0.5
ARROW_WIDTH = 0.75
# A dot's radius, as a share of the spacing between points.
DOT_RADIUS = 0.09
# A triangle with a side longer than this many point spacings bridges a gap
# in the mask - between the face and an ear standing apart, or across a
# concave edge - rather than covering face, and is left out.
LONGEST_SIDE = 2.5


def triangles(points):
    """Delaunay triangles over `points`, without those that bridge gaps."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return np.zeros((0, 3), int)
    simplices = Delaunay(points).simplices
    corners = points[simplices]
    sides = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    return simplices[sides.max(axis=1) <= LONGEST_SIDE * spacing_of(points)]


def _caption(image, text, scale):
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.6 * max(scale, 0.5)
    weight = max(1, int(round(size * 2)))
    (width, height), base = cv2.getTextSize(text, font, size, weight)
    pad = int(round(8 * max(scale, 0.5)))
    box = image[0:height + base + 2 * pad, 0:width + 2 * pad]
    box[...] = (box * 0.35).astype(np.uint8)
    cv2.putText(image, text, (pad, pad + height), font, size, (255, 255, 255), weight,
                cv2.LINE_AA)


class _Marks:
    """A transparent layer `SUPERSAMPLE` times the picture's size, drawn on
    and then laid over the picture, scaled down. The photograph itself is
    never resampled; only the marks are."""

    def __init__(self, image):
        self.image = image
        height, width = image.shape[:2]
        self.colour = np.zeros((height * SUPERSAMPLE, width * SUPERSAMPLE, 3), np.uint8)
        self.alpha = np.zeros((height * SUPERSAMPLE, width * SUPERSAMPLE), np.uint8)

    @staticmethod
    def _at(points):
        return np.round(np.asarray(points, float) * SUPERSAMPLE).astype(np.int32)

    @staticmethod
    def _width(pixels):
        return max(1, int(round(pixels * SUPERSAMPLE)))

    def lines(self, polygons, colour, width):
        polygons = [self._at(polygon) for polygon in polygons]
        for layer, value in ((self.colour, colour), (self.alpha, 255)):
            cv2.polylines(layer, polygons, True, value, self._width(width), cv2.LINE_AA)

    def arrow(self, start, end, colour, width):
        start, end = tuple(self._at(start)), tuple(self._at(end))
        for layer, value in ((self.colour, colour), (self.alpha, 255)):
            cv2.arrowedLine(layer, start, end, value, self._width(width), cv2.LINE_AA,
                            tipLength=0.3)

    def cross(self, centre, colour, size, width):
        x, y = centre
        strokes = [[(x - size, y - size), (x + size, y + size)],
                   [(x - size, y + size), (x + size, y - size)]]
        strokes = [self._at(stroke) for stroke in strokes]
        for layer, value in ((self.colour, colour), (self.alpha, 255)):
            cv2.polylines(layer, strokes, False, value, self._width(width), cv2.LINE_AA)

    def dot(self, centre, colour, radius):
        centre, radius = tuple(self._at(centre)), self._width(radius)
        for layer, value in ((self.colour, colour), (self.alpha, 255)):
            cv2.circle(layer, centre, radius, value, -1, cv2.LINE_AA)

    def lay(self):
        height, width = self.image.shape[:2]
        colour = cv2.resize(self.colour, (width, height), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(self.alpha, (width, height), interpolation=cv2.INTER_AREA)
        alpha = alpha[..., None].astype(float) / 255.0
        # The colour layer is dark where nothing was drawn, and averaging
        # takes that darkness in at every edge; dividing by the coverage
        # takes it out again.
        colour = colour / np.maximum(alpha, 1e-6)
        self.image[...] = np.clip(self.image * (1 - alpha) + colour * alpha, 0, 255).astype(np.uint8)


def views(drift, frame, fit="similarity", points=True, mesh=True, exaggeration=1,
          dropped=True, floor=NOISE_FLOOR):
    """`(before_view, after_view)`, uint8 RGB, both `frame` in size.

    The after is moved by the `fit` alone. `dropped` marks the points that
    failed the flow's round trip.
    """
    after_px, before_px, _ = chosen(drift.record, drift.ears)
    source = geometry.into_frame(after_px, drift.after_size, frame)
    target = geometry.into_frame(before_px, drift.before_size, frame)
    width, height = (int(round(side)) for side in frame)

    warp = Warp(source, target, geometry.fit(source, target, fit), frame)
    after_image = warp.render(drift.after, 0.0)[..., :3].copy()
    landed = geometry.apply(warp.matrix, source)
    title = f"after, {fit} fit"

    # The before brought into the frame the same way the after is: a cover
    # fit, which for the before's own shape is a plain resize.
    scale, offset = geometry.cover(drift.before_size, frame)
    resample = np.array([[scale, 0.0, -offset[0]], [0.0, scale, -offset[1]]])
    before_image = cv2.warpAffine(drift.before, resample, (width, height),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    left = geometry.residuals(landed, target)
    quiet = left <= floor

    line_scale = max(width, height) / 1000.0
    radius = spacing_of(target) * DOT_RADIUS
    on_before, on_after = _Marks(before_image), _Marks(after_image)

    if mesh:
        faces = triangles(target)
        on_before.lines([target[face] for face in faces], MESH_BEFORE, MESH_WIDTH * line_scale)
        on_after.lines([landed[face] for face in faces], MESH_AFTER, MESH_WIDTH * line_scale)
    if points:
        for centre in target:
            on_before.dot(centre, BEFORE, radius)
        for start, end, still in zip(landed, landed + exaggeration * (target - landed), quiet):
            on_after.arrow(start, end, QUIET if still else AFTER, ARROW_WIDTH * line_scale)
        for centre, still in zip(landed, quiet):
            on_after.dot(centre, QUIET if still else AFTER, radius)
    gaps = geometry.into_frame(dropped_points(drift.record, drift.ears),
                               drift.before_size, frame)
    if dropped:
        for centre in gaps:
            on_before.cross(centre, DROPPED, radius * 1.4, ARROW_WIDTH * line_scale)
    on_before.lay()
    on_after.lay()

    marks = f"arrows x{exaggeration}" if points else ""
    _caption(before_image, f"before, {len(target)} points"
             + (f", {len(gaps)} dropped (flow did not come back)" if dropped and len(gaps) else ""),
             line_scale)
    _caption(after_image, f"{title}{', ' + marks if marks else ''}", line_scale)
    return before_image, after_image
