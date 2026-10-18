"""Measuring how far an edit moved a face, and moving it back.

The steps, each in the module that owns it:

1. A face mask on each side - `models.FaceMasker`, Grounded SAM 2 asked for
   "face" and "ear".
2. Each side cropped to its own face and resampled into one working frame,
   so the optical flow only has to see what the edit changed - not a resize,
   a reframing or a move (`flow_sampling`).
3. RAFT's flow both ways; grid points inside the before mask whose round
   trip comes back are the matched points (`flow_sampling.sample`).
4. Both sides brought into the before image's frame, a similarity and an
   affine fitted, and what each leaves at every point is the drift
   (`geometry`).
5. Repair: the flat fit, then the smooth warp, from the same points
   (`smooth_warp`).

The models are passed in, so everything here runs without a GPU in tests.
"""
from dataclasses import dataclass, field

import numpy as np

from . import geometry
from .flow_sampling import (box_transform, crop_points, crop_to_working, face_box,
                            outcomes, region_contour, sample, tags, to_image_pixels)
from .smooth_warp import render_repaired

PROVIDER = "raft"
# The median residual RAFT reports on a pure resample, as a percentage of face
# size: below it, an apparent shift is the measurement's own noise. Measured
# on a 25-pair null test (every face resampled 512 -> 1414 px, so the true
# shift is exactly zero) under the same settings as here.
NOISE_FLOOR = 0.05
# Divisible by 8, which RAFT's feature pyramid requires.
WORKING_SIZE = 512
# Grid step in working-frame pixels: about a thousand points over the crop.
GRID_STEP = 16
# Forward-backward agreement, in working-frame pixels, for a point to be kept.
# RAFT reports no confidence of its own, so this is the whole of it: run the
# flow both ways and see whether it comes back. Where it does not, the flow is
# guessing - an occlusion, or texture the edit rewrote.
CONSISTENCY_PX = 1.5
# The largest side a repair is drawn at.
MAX_SIDE = 6000


@dataclass
class Region:
    """One side's face mask, as `models.FaceMasker` finds it."""

    mask: np.ndarray            # (h, w) uint8, 1 on the face (and joined ears)
    outline: np.ndarray         # (n, 2) the mask's outer boundary, in order
    parts: dict                 # {"face": outline, "ears": [{"points", "joined"}]}
    face_score: float = 0.0
    faces_found: int = 0
    ears_found: int = 0
    ears_joined: int = 0


@dataclass
class FlowRecord:
    """Every grid point the flow was asked about, and what became of it."""

    points: np.ndarray          # (n, 5) origin x, y, landed x, y, round-trip error
    outcomes: list
    tags: list                  # face, ear or outside, in the before frame
    before_box: tuple
    after_box: tuple
    working_size: int = WORKING_SIZE
    consistency_px: float = CONSISTENCY_PX


@dataclass
class Fit:
    mode: str
    matrix: np.ndarray          # 2x3, after onto before, in the before frame
    residuals: np.ndarray       # % of face size, one per point
    median: float
    p90: float
    highest: float
    above_noise: float          # share of points beyond the noise floor
    transform: dict


@dataclass
class Drift:
    """What one pair measured: the points, the fits, and the pictures' inputs.

    Carries the two images as well, so that a repair can be made from it with
    nothing else wired in.
    """

    before: np.ndarray          # (h, w, 3) uint8 RGB
    after: np.ndarray
    before_region: Region
    after_region: Region
    record: FlowRecord
    after_points: np.ndarray    # (n, 2) chosen points, in the before frame
    before_points: np.ndarray
    confidence: np.ndarray
    ears: bool
    fits: dict = field(default_factory=dict)

    @property
    def before_size(self):
        return (self.before.shape[1], self.before.shape[0])

    @property
    def after_size(self):
        return (self.after.shape[1], self.after.shape[0])


def sample_flow(flows, before, after, before_region, after_region,
                working_size=WORKING_SIZE, grid_step=GRID_STEP,
                consistency_px=CONSISTENCY_PX):
    """Crop each side to its face, run the flow both ways, tag every point.

    `flows(first, second)` takes two working frames and returns the forward
    and the backward flow, each `(2, size, size)`.

    The crop takes in the face and every ear found, joined or not, so that
    "ears in" and "ears out" are the same run and only the choice of points
    differs.
    """
    frames, transforms, cut = {}, {}, {}
    for side, rgb, region in (("before", before, before_region),
                              ("after", after, after_region)):
        height, width = rgb.shape[:2]
        cut[side] = face_box(crop_points(region.outline, region.parts), width, height)
        frames[side], transforms[side] = crop_to_working(rgb, cut[side], working_size)

    # The grid lives in the *before* working frame - `sample` walks forward
    # from the before side - so the before outlines are the ones that decide
    # which part of the mask a point is on.
    def contour(points):
        return region_contour(points, transforms["before"], convex=False)

    parts = before_region.parts
    region = contour(before_region.outline)
    face = contour(parts["face"])
    apart = [contour(ear["points"]) for ear in parts.get("ears", []) if not ear["joined"]]

    flow_ab, flow_ba = flows(frames["before"], frames["after"])
    sampled = sample(flow_ab, flow_ba, grid_step, consistency_px)
    return FlowRecord(
        points=np.column_stack([sampled.origin, sampled.landed, sampled.error]),
        outcomes=outcomes(sampled),
        tags=tags(sampled.origin, region, face, apart),
        before_box=cut["before"], after_box=cut["after"],
        working_size=working_size, consistency_px=consistency_px)


def chosen(record, ears=True):
    """The points that feed a fit, in image pixels: after, before, confidence.

    A point counts when it is on the face (or an ear, if `ears`) and its round
    trip came back.
    """
    use = np.array([outcome.startswith("kept") and (tag == "face" or (tag == "ear" and ears))
                    for outcome, tag in zip(record.outcomes, record.tags)], bool)
    if not use.any():
        raise ValueError("no point on the face passed the flow's round-trip check; "
                         "the two images may share no recoverable correspondence")
    points = np.asarray(record.points, float)[use]
    before = to_image_pixels(points[:, 0:2], box_transform(record.before_box, record.working_size))
    after = to_image_pixels(points[:, 2:4], box_transform(record.after_box, record.working_size))
    confidence = 1.0 - np.clip(points[:, 4] / record.consistency_px, 0.0, 1.0)
    return after, before, confidence


def dropped(record, ears=True):
    """The points on the face (and ears, if `ears`) that failed the round
    trip, in before-image pixels. Where they sit is a gap in the grid."""
    use = np.array([outcome.startswith("dropped") and (tag == "face" or (tag == "ear" and ears))
                    for outcome, tag in zip(record.outcomes, record.tags)], bool)
    points = np.asarray(record.points, float)[use]
    return to_image_pixels(points[:, 0:2], box_transform(record.before_box, record.working_size))


def fit_drift(after_points, before_points, mode, noise_floor=NOISE_FLOOR):
    """One flat fit of after onto before, and what it leaves at every point."""
    matrix = geometry.fit(after_points, before_points, mode)
    each = geometry.residuals(geometry.apply(matrix, after_points), before_points)
    return Fit(mode=mode, matrix=matrix, residuals=each,
               median=float(np.median(each)), p90=float(np.percentile(each, 90)),
               highest=float(each.max()),
               above_noise=float((each > noise_floor).mean()),
               transform=geometry.describe(matrix, after_points, before_points))


def measure(before, after, masker, flows, ears=True):
    """Everything about one pair. `before` and `after` are uint8 RGB arrays.

    `masker(rgb)` returns a `Region`; `flows` is as `sample_flow` takes it.
    """
    before_region, after_region = masker(before), masker(after)
    record = sample_flow(flows, before, after, before_region, after_region)
    after_px, before_px, confidence = chosen(record, ears)

    # Both sides in the before image's own frame: the before points are
    # already there, and the after points are brought over by the cover fit.
    # The fits, their figures and the map are all read in that frame.
    frame = (before.shape[1], before.shape[0])
    after_points = geometry.into_frame(after_px, (after.shape[1], after.shape[0]), frame)
    before_points = geometry.into_frame(before_px, frame, frame)

    drift = Drift(before=before, after=after, before_region=before_region,
                  after_region=after_region, record=record,
                  after_points=after_points, before_points=before_points,
                  confidence=confidence, ears=ears)
    drift.fits = {mode: fit_drift(after_points, before_points, mode)
                  for mode in geometry.FITS}
    return drift


def repair_frame(drift, max_side=MAX_SIDE):
    """The frame a repair is drawn in: the before's shape, enlarged until the
    after image is sampled about one to one, so the after's detail is kept.

    It lines up with the before pixel for pixel, only bigger. An after that is
    smaller on the face than the before comes out smaller, not upscaled.
    """
    frame = drift.before_size
    width, height = drift.after_size
    shown = max(frame[0] / width, frame[1] / height) * drift.fits["similarity"].transform["scale"]
    enlarge = min(1.0 / shown, max_side / max(frame))
    return tuple(float(round(side * enlarge)) for side in frame)


def repair(drift, strength=1.0, frame=None):
    """The after image with its features moved back, and where it is covered.

    The flat similarity fit, then the smooth warp on top, from the same points
    the measurement chose. `strength` is how far along the warp goes: 0 is the
    flat fit alone. Returns `(rgb, covered)`, uint8 and bool, in `frame`.
    """
    frame = frame or repair_frame(drift)
    record = drift.record
    after_px, before_px, _ = chosen(record, drift.ears)
    source = geometry.into_frame(after_px, drift.after_size, frame)
    target = geometry.into_frame(before_px, drift.before_size, frame)
    matrix = geometry.fit(source, target, "similarity")
    rendered = render_repaired(drift.after, source, target, matrix, frame,
                               strength=strength)
    return rendered[..., :3], rendered[..., 3] > 0
