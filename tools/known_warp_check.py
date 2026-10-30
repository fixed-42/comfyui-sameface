#!/usr/bin/env python3
"""Check Measure and Repair against a truth we chose.

The Lincoln before is moved by known warps, with and without a change of
colour, tone and sharpness, and put through the real models. Run it with
ComfyUI's Python, from the pack's folder:

    python tools/known_warp_check.py

For each case it prints the drift the warp really left against what Measure
read, how much of the move was read where the warp moved most, the share of
points kept, what the repair really left over the whole face (worked out from
the warp, with no flow), what a fresh measurement says the repair left, and
the pixels against the same picture unwarped. See `sameface/known_warp.py` for
what this cannot show.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sameface import geometry  # noqa: E402
from sameface.known_warp import (NO_WARP, bump, face_points, left_by_repair,  # noqa: E402
                                 pixel_gap, restyle, score, tracked_drift, widen)
from sameface.measure import chosen, dropped, measure, repair  # noqa: E402
from sameface.models import FaceMasker, Flow  # noqa: E402

BEFORE = Path(__file__).resolve().parent.parent / "example_workflows/input/lincoln_before.png"
# The eye on the left of the picture, in the before's pixels.
EYE = (539.0, 489.0)
# How far the eye moves, in % of face size, and how much the head widens, in
# %. A real edit drifts broadly, so the widenings come down to the small
# drifts SameFace reports; the eye is one feature moved alone.
EYE_MOVES = (0.1, 0.3, 1.0)
WIDENINGS = (3.7, 1.0, 0.3, 0.1)
# Fixed before the first run.
MARKS = {"read": (0.8, 1.2), "kept": 0.9, "left": 0.25}


def cases(centre, radii, size):
    yield "no warp", NO_WARP
    for rate in WIDENINGS:
        yield f"head {rate}% wider", widen(centre, radii, rate / 100.0)
    for move in EYE_MOVES:
        yield f"eye up {move}%", bump(EYE, 0.1 * size, (0.0, -move / 100.0 * size))


def main():
    before = np.asarray(Image.open(BEFORE).convert("RGB"))
    masker, flows = FaceMasker(), Flow()
    region = masker(before)
    plain = measure(before, before.copy(), masker, flows, before_region=region)
    size = geometry.point_set_size(plain.before_points)
    centre = plain.before_points.mean(axis=0)
    radii = np.ptp(plain.before_points, axis=0) / 2.0
    points = face_points(region.mask > 0)
    print(f"face size {size:.1f} px, {len(plain.before_points)} points, "
          f"{len(points)} checked over the face\n")

    print(f"{'case':24}{'true':>8}{'read':>8}{'share':>7}{'error':>8}{'kept':>7}"
          f"{'left':>8}{'of':>8}{'lk':>8}{'again':>8}{'px edit':>9}{'px rep':>8}")
    for restyled in (False, True):
        like = restyle(before) if restyled else before
        for name, warp in cases(centre, radii, size):
            after = warp.render(before)
            after = restyle(after) if restyled else after
            drift = measure(before, after, masker, flows, before_region=region)
            result = score(drift.after_points, drift.before_points, warp)
            lost = len(dropped(drift.record, drift.ears))
            kept = len(drift.before_points) / (len(drift.before_points) + lost)

            after_px, before_px, _ = chosen(drift.record, drift.ears)
            left, started = left_by_repair(after_px, before_px, warp, points, drift.after_size)
            mended, _ = repair(drift)
            again = measure(before, mended, masker, flows, before_region=region)

            pose = geometry.fit(before_px, after_px, "similarity")
            target = cv2.warpAffine(like, pose, drift.after_size, flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            inside = cv2.warpAffine((region.mask > 0).astype(np.uint8), pose,
                                    drift.after_size, flags=cv2.INTER_NEAREST) > 0

            share = "-" if result.read is None else f"{result.read:.2f}"
            label = f"{name}{' +style' if restyled else ''}"
            print(f"{label:24}{result.true['similarity']:8.3f}"
                  f"{result.measured['similarity']:8.3f}{share:>7}{result.point_error:8.3f}"
                  f"{kept:7.1%}{np.median(left):8.3f}{np.median(started):8.3f}"
                  f"{np.median(tracked_drift(target, mended, geometry.apply(pose, points))):8.3f}"
                  f"{again.fits['similarity'].median:8.3f}"
                  f"{pixel_gap(after, target, inside):9.2f}"
                  f"{pixel_gap(mended, target, inside):8.2f}", flush=True)

    print(f"\n% of face size, similarity fit, medians. true: the drift the warp left. read: "
          f"what Measure read. share: of the move, read where it moved most (mark "
          f"{MARKS['read'][0]}-{MARKS['read'][1]}). error: each point against its truth. kept: "
          f"points that came back (mark {MARKS['kept']:.0%}). left: what the repair really left "
          f"over the whole face, of what it started with (mark under {MARKS['left']:.0%}). lk: the "
          f"same by Lucas-Kanade tracking on the pixels, no RAFT. "
          f"again: what a fresh measurement says it left. px: mean difference inside the face "
          f"from the same picture unwarped, for the edit and the repair.")


if __name__ == "__main__":
    main()
