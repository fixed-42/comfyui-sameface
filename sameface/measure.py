"""Measuring how far an edit moved a face, and moving it back.

The steps, each in the module that owns it:

1. A face mask on each side - `models.FaceMasker`, Grounded SAM 2 asked for
   "face" and "ear".
2. Each side cropped to its own face and resampled into one working frame,
   so the optical flow only has to see what the edit changed - not a resize,
   a reframing or a move (`flow_sampling`).
3. RAFT's flow both ways; grid points inside the before mask whose round
   trip comes back are the matched points (`flow_sampling.sample`).
   Then a second pass: the after is cut again where a similarity fit through
   those points puts the before's crop - turned and scaled with it - and the
   flow runs once more. Box crops leave a turn and a little scale for RAFT to
   follow, and it reads part of that as drift: an after merely moved onto
   the before measured 35-45% lower than the same after as the edit framed
   it. Lined up first, the figure no longer depends on the framing.
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
from .flow_sampling import (apply_affine, box_matrix, compose, crop_affine, crop_points,
                            crop_to_working, face_box, invert, outcomes, region_contour,
                            sample, tags)
from .smooth_warp import render_repaired

PROVIDER = "raft"
# Below this, as a percentage of face size, an apparent shift is the
# measurement's own noise. Declared by `calibration.declared_floor` from
# `tools/calibrate_noise_floor.py` on 25 null pairs (each face resampled to
# 512 and 1414 px and saved as a JPEG, so the true shift is zero): a median of
# 0.0033% with two passes, so 0.01%. One pass measured 0.0167%, so 0.05%.
# Run the tool again whenever the measurement changes.
NOISE_FLOOR = 0.01
# Flow runs per pair: the second lines the after up with the before first.
PASSES = 2
# Divisible by 8, which RAFT's feature pyramid requires.
WORKING_SIZE = 512
# Grid step in working-frame pixels: about a thousand points over the crop.
GRID_STEP = 16
# Forward-backward agreement, in working-frame pixels, for a point to be kept.
# RAFT reports no confidence of its own, so this is the whole of it: run the
# flow both ways and see whether it comes back. Where it does not, the flow is
# guessing - an occlusion, or texture the edit rewrote.
CONSISTENCY_PX = 1.5
# The largest side a repair is drawn at, in the before's frame.
MAX_SIDE = 6000
# Whose framing a repair keeps: the after's own (the edit's composition, only
# the face's features moved), or the before's (the whole after lined up with
# the before, as the measurement sees it).
FRAMINGS = ("after", "before")


@dataclass
class Region:
    """One side's face mask, as `models.FaceMasker` finds it or as it was
    supplied (`regions.region_from_mask`)."""

    mask: np.ndarray            # (h, w) uint8, 1 on the face (and joined ears)
    outline: np.ndarray         # (n, 2) the mask's outer boundary, in order
    parts: dict                 # {"face": outline, "ears": [{"points", "joined"}]}
    face_score: float = 0.0
    faces_found: int = 0
    ears_found: int = 0
    ears_joined: int = 0
    supplied: bool = False      # wired in, not found here: all of it is face
    pieces_dropped: int = 0     # supplied pieces left out, the largest kept
    box_shaped: bool = False    # supplied, and near enough a rectangle


@dataclass
class FlowRecord:
    """Every grid point the flow was asked about, and what became of it."""

    points: np.ndarray          # (n, 5) origin x, y, landed x, y, round-trip error
    outcomes: list
    tags: list                  # face, ear or outside, in the before frame
    before_map: np.ndarray      # 2x3, before working frame to before image pixels
    after_map: np.ndarray       # 2x3, after working frame to after image pixels
    working_size: int = WORKING_SIZE
    consistency_px: float = CONSISTENCY_PX
    passes: int = 1


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
                consistency_px=CONSISTENCY_PX, after_map=None):
    """Crop each side to its face, run the flow both ways, tag every point.

    `flows(first, second)` takes two working frames and returns the forward
    and the backward flow, each `(2, size, size)`.

    The crop takes in the face and every ear found, joined or not, so that
    "ears in" and "ears out" are the same run and only the choice of points
    differs. `after_map` (2x3, working frame to after pixels) cuts the after
    by that instead of by its own box: the second pass.
    """
    frames, transforms, maps = {}, {}, {}
    for side, rgb, region in (("before", before, before_region),
                              ("after", after, after_region)):
        if side == "after" and after_map is not None:
            frames[side], maps[side] = crop_affine(rgb, after_map, working_size), after_map
            continue
        height, width = rgb.shape[:2]
        cut = face_box(crop_points(region.outline, region.parts), width, height)
        frames[side], transforms[side] = crop_to_working(rgb, cut, working_size)
        maps[side] = box_matrix(cut, working_size)

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
        before_map=maps["before"], after_map=maps["after"],
        working_size=working_size, consistency_px=consistency_px,
        passes=1 if after_map is None else 2)


def _used(record, ears):
    """Which grid points feed a fit: on the face (or an ear, if `ears`), and
    their round trip came back."""
    return np.array([outcome.startswith("kept") and (tag == "face" or (tag == "ear" and ears))
                     for outcome, tag in zip(record.outcomes, record.tags)], bool)


def chosen_parts(record, ears=True):
    """`face` or `ear` for each point `chosen` returns, in the same order."""
    return [tag for tag, use in zip(record.tags, _used(record, ears)) if use]


def chosen(record, ears=True):
    """The points that feed a fit, in image pixels: after, before, confidence.

    A point counts when it is on the face (or an ear, if `ears`) and its round
    trip came back.
    """
    use = _used(record, ears)
    if not use.any():
        raise ValueError("no point on the face passed the flow's round-trip check; "
                         "the two images may share no recoverable correspondence")
    points = np.asarray(record.points, float)[use]
    before = apply_affine(points[:, 0:2], record.before_map)
    after = apply_affine(points[:, 2:4], record.after_map)
    confidence = 1.0 - np.clip(points[:, 4] / record.consistency_px, 0.0, 1.0)
    return after, before, confidence


def dropped(record, ears=True):
    """The points on the face (and ears, if `ears`) that failed the round
    trip, in before-image pixels. Where they sit is a gap in the grid."""
    use = np.array([outcome.startswith("dropped") and (tag == "face" or (tag == "ear" and ears))
                    for outcome, tag in zip(record.outcomes, record.tags)], bool)
    points = np.asarray(record.points, float)[use]
    return apply_affine(points[:, 0:2], record.before_map)


def fit_drift(after_points, before_points, mode, noise_floor=NOISE_FLOOR):
    """One flat fit of after onto before, and what it leaves at every point."""
    matrix = geometry.fit(after_points, before_points, mode)
    each = geometry.residuals(geometry.apply(matrix, after_points), before_points)
    return Fit(mode=mode, matrix=matrix, residuals=each,
               median=float(np.median(each)), p90=float(np.percentile(each, 90)),
               highest=float(each.max()),
               above_noise=float((each > noise_floor).mean()),
               transform=geometry.describe(matrix, after_points, before_points))


def measure(before, after, masker, flows, ears=True, before_region=None, after_region=None,
            passes=PASSES):
    """Everything about one pair. `before` and `after` are uint8 RGB arrays.

    `masker(rgb)` returns a `Region`, and is asked only for a side whose
    region was not given; `flows` is as `sample_flow` takes it. `passes=1`
    is the measurement without the second pass.
    """
    before_region = before_region if before_region is not None else masker(before)
    after_region = after_region if after_region is not None else masker(after)
    record = sample_flow(flows, before, after, before_region, after_region)
    if passes > 1:
        # Every kept point, ears too, so "ears in" and "ears out" stay one run.
        after_px, before_px, _ = chosen(record, ears=True)
        pose = geometry.fit(after_px, before_px, "similarity")
        record = sample_flow(flows, before, after, before_region, after_region,
                             after_map=compose(invert(pose), record.before_map))
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


def repair(drift, strength=1.0, frame=None, framing="after"):
    """The after image with its features moved back, and where it is covered.

    `framing="after"` keeps the after as the edit framed it, at its own size:
    the before's points are carried into the after by the after's own pose (a
    similarity fit, before onto after), and the smooth warp moves each feature
    to where the before has it in that pose. Beyond the face the warp eases
    to nothing, so the rest of the after is untouched and all of it is covered.

    `framing="before"` lines the whole after up with the before: the flat
    similarity fit, then the smooth warp on top, drawn in `frame` (by default
    `repair_frame`). Where the after does not reach is uncovered.

    `strength` is how far along the warp goes: 0 is the after unchanged, or
    the flat fit alone. Returns `(rgb, covered)`, uint8 and bool.
    """
    record = drift.record
    after_px, before_px, _ = chosen(record, drift.ears)
    if framing == "after":
        pose = geometry.fit(before_px, after_px, "similarity")
        rendered = render_repaired(drift.after, after_px, geometry.apply(pose, before_px),
                                   np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                                   drift.after_size, strength=strength)
        return rendered[..., :3], rendered[..., 3] > 0
    if framing != "before":
        raise ValueError(f"framing is one of {FRAMINGS}, not {framing!r}")
    frame = frame or repair_frame(drift)
    source = geometry.into_frame(after_px, drift.after_size, frame)
    target = geometry.into_frame(before_px, drift.before_size, frame)
    matrix = geometry.fit(source, target, "similarity")
    rendered = render_repaired(drift.after, source, target, matrix, frame,
                               strength=strength)
    return rendered[..., :3], rendered[..., 3] > 0
