#!/usr/bin/env python3
"""Measure the noise floor: a folder of face images, each measured against
itself resampled (512 and 1414 pixels on the long side, from the same
source), where the true drift is zero.

Run it with ComfyUI's Python, which has the models, from the pack's folder:

    python tools/calibrate_noise_floor.py <folder of face images> [--passes 1|2]

It prints each image's similarity median, then the median, p90 and largest
over them and the floor `sameface.calibration.declared_floor` gives. If that
differs from `NOISE_FLOOR` in `sameface/measure.py`, the measurement has
changed and the floor should be declared again. One face per image.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sameface.calibration import declared_floor, null_pair  # noqa: E402
from sameface.measure import NOISE_FLOOR, PASSES, measure  # noqa: E402
from sameface.models import FaceMasker, Flow  # noqa: E402

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--passes", type=int, default=PASSES, choices=(1, 2))
    args = parser.parse_args()
    paths = sorted(path for path in args.folder.iterdir() if path.suffix.lower() in IMAGES)
    if not paths:
        sys.exit(f"no images in {args.folder}")

    masker, flows = FaceMasker(), Flow()
    medians = []
    for path in paths:
        small, large = null_pair(np.asarray(Image.open(path).convert("RGB")))
        drift = measure(small, large, masker, flows, passes=args.passes)
        medians.append(drift.fits["similarity"].median)
        print(f"{medians[-1]:.4f}%  {path.name}", flush=True)

    values = np.array(medians)
    floor = declared_floor(values)
    print(f"\n{len(values)} null pairs, {args.passes} pass(es), similarity fit, % of face size")
    print(f"median {np.median(values):.4f}  p90 {np.percentile(values, 90):.4f}  "
          f"largest {values.max():.4f}")
    print(f"declared floor {floor:g}% (twice the median, up to a 1-2-5 step); "
          f"NOISE_FLOOR is {NOISE_FLOOR:g}%{'' if floor == NOISE_FLOOR else ' - declare it again'}")


if __name__ == "__main__":
    main()
