"""Known warps: a pair whose true drift is chosen, not measured.

A null pair says how far the measurement moves when nothing moved. It cannot
say whether a real move is read at its true size, or whether a repair puts a
feature back. Here the after is the before moved by a warp we chose, so the
truth is known at every point: `score` sets what Measure read against it, and
`left_by_repair` works out where the repair's own warp puts each point, by
arithmetic, with no second run of the flow. `tools/known_warp_check.py` runs
this through the real models.

**What it cannot show.** A warped copy keeps its texture, which is the easy
case for optical flow: a real edit also redraws it. `restyle` changes colour,
tone and sharpness to close part of that gap, not all of it.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry
from .calibration import JPEG_QUALITY
from .geometry import FITS
from .measure import fit_drift
from .smooth_warp import Warp

# The warp is given forwards, before to after; a render needs it backwards.
# It is smooth and small, so stepping settles in a few rounds.
ROUNDS = 8
# Points that moved at least this share of the most any point moved are the
# ones `Score.read` is taken over: elsewhere there is little move to read.
MOVED_SHARE = 0.5
IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
LK_WINDOW = (31, 31)


class KnownWarp:
    """`offset(points)` says how far each before point moves, in pixels."""

    def __init__(self, offset):
        self.offset = offset

    def forward(self, points):
        """Where before points land in the after."""
        points = np.asarray(points, float)
        return points + self.offset(points)

    def render(self, rgb):
        """`rgb` as the after: each pixel fetched from where the warp brought
        it from."""
        height, width = rgb.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=float),
                                     np.arange(height, dtype=float))
        wanted = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        if not np.any(self.offset(wanted)):
            return rgb.copy()
        came_from = wanted.copy()
        for _ in range(ROUNDS):
            came_from = wanted - self.offset(came_from)
        came_from = came_from.reshape(height, width, 2).astype(np.float32)
        return cv2.remap(rgb, came_from[..., 0], came_from[..., 1], cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)


NO_WARP = KnownWarp(lambda points: np.zeros_like(np.asarray(points, float)))


def bump(centre, sigma, vector):
    """One region moved by `vector` at `centre`, fading as a Gaussian."""
    centre, vector = np.asarray(centre, float), np.asarray(vector, float)

    def offset(points):
        away = ((np.asarray(points, float) - centre) ** 2).sum(axis=1)
        return np.exp(-away / (2.0 * sigma ** 2))[:, None] * vector

    return KnownWarp(offset)


def widen(centre, radii, rate):
    """Stretched across by `rate` about `centre`, whole inside the ellipse
    `radii` and eased to nothing by twice as far."""
    centre, radii = np.asarray(centre, float), np.asarray(radii, float)

    def offset(points):
        across = np.asarray(points, float) - centre
        reach = np.clip(2.0 - np.hypot(*(across / radii).T), 0.0, 1.0)
        moved = np.zeros_like(across)
        moved[:, 0] = rate * across[:, 0] * reach * reach * (3.0 - 2.0 * reach)
        return moved

    return KnownWarp(offset)


def restyle(rgb):
    """A change that moves nothing: warmer, softer, and through a JPEG."""
    tone = np.power(rgb.astype(np.float32) / 255.0, np.array([0.9, 1.0, 1.15], np.float32))
    soft = cv2.GaussianBlur(tone, (0, 0), 0.8)
    styled = (soft * 255.0).round().clip(0, 255).astype(np.uint8)
    _, data = cv2.imencode(".jpg", styled[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@dataclass
class Score:
    true: dict                  # fit -> the median drift the warp really left
    measured: dict              # fit -> the median drift Measure read
    point_error: float          # median distance from each point's truth, % of size
    read: float                 # share of the move read where it moved most; None if none


def score(after_points, before_points, warp):
    """What Measure read, against the truth. All in the before frame."""
    after_points, before_points = np.asarray(after_points, float), np.asarray(before_points, float)
    truth = warp.forward(before_points)
    moved = truth - before_points
    size = np.hypot(*moved.T)
    read = None
    if size.max() > 0:
        most = size >= MOVED_SHARE * size.max()
        seen = ((after_points - before_points)[most] * moved[most]).sum(axis=1)
        read = float(np.median(seen / (size[most] ** 2)))
    return Score(true={fit: fit_drift(truth, before_points, fit).median for fit in FITS},
                 measured={fit: fit_drift(after_points, before_points, fit).median
                           for fit in FITS},
                 point_error=float(np.median(geometry.residuals(after_points, truth))),
                 read=read)


def left_by_repair(after_px, before_px, warp, points, after_size):
    """How far `points` of the before lie from where they belong, in % of
    size, after the repair and before it.

    The repair built from the measured `after_px` and `before_px`, as
    `measure.repair` builds it in the after's framing, is applied to where
    the warp truly put each point. No flow runs here: what comes back is the
    repair's own geometry against the known truth. It is drift as Measure
    defines it, with the move, scale and turn of the whole face taken out.
    """
    pose = geometry.fit(before_px, after_px, "similarity")
    repair = Warp(after_px, geometry.apply(pose, before_px), IDENTITY, after_size)
    truth = warp.forward(points)
    return (fit_drift(repair.moved(truth), points, "similarity").residuals,
            fit_drift(truth, points, "similarity").residuals)


def tracked_drift(first, second, points):
    """The drift from `first` to `second` at `points`, in % of size, by
    Lucas-Kanade tracking: classical, with no RAFT in it, so it can check a
    measurement RAFT made. It needs the two images to look alike, which only a
    pair with a known warp gives: a real edit redraws what it would track."""
    from_grey, to_grey = (cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in (first, second))
    start = np.asarray(points, np.float32).reshape(-1, 1, 2)
    end, found, _ = cv2.calcOpticalFlowPyrLK(
        from_grey, to_grey, start, None, winSize=LK_WINDOW, maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4))
    found = found.ravel() == 1
    return fit_drift(end.reshape(-1, 2)[found], start.reshape(-1, 2)[found], "similarity").residuals


def face_points(mask, step=8):
    """A regular grid over a boolean mask, in (x, y): the whole face, not only
    the points the flow kept, so a feature it dropped still counts."""
    ys, xs = np.nonzero(mask)
    keep = (xs % step == 0) & (ys % step == 0)
    return np.column_stack([xs[keep], ys[keep]]).astype(float)


def pixel_gap(first, second, mask):
    """The mean difference between two images inside `mask`, on 0-255."""
    return float(np.abs(first[mask].astype(float) - second[mask].astype(float)).mean())
