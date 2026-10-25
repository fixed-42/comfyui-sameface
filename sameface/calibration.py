"""The noise floor: what Measure reports where the true drift is zero.

A null pair is one face at two sizes, both resampled from the same source -
the long side at 512 and at 1414 pixels, a scale of exactly 2.76172 - and
each saved as a JPEG at quality 92, as the first null pairs were: clean
pixels measure about half as noisy, and no real image arrives clean.
Whatever drift Measure reports on it is the measurement's own.
`tools/calibrate_noise_floor.py` measures a folder of faces this way and
prints the floor `declared_floor` gives; `measure.NOISE_FLOOR` is that value.
"""
import math

import cv2
import numpy as np

NULL_SIDES = (512, 1414)
JPEG_QUALITY = 92
# Declared floors are rounded up to the next of these, times a power of ten.
STEPS = (1, 2, 5, 10)


def null_pair(rgb, sides=NULL_SIDES, quality=JPEG_QUALITY):
    """`rgb` with its long side at each of `sides`, each resampled from it
    and passed through a JPEG at `quality` (None for clean pixels)."""
    height, width = rgb.shape[:2]
    pair = []
    for side in sides:
        scale = side / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1
                             else cv2.INTER_CUBIC)
        pair.append(resized if quality is None else _jpeg(resized, quality))
    return tuple(pair)


def _jpeg(rgb, quality):
    _, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def declared_floor(null_medians):
    """Twice the median of the null pairs' medians, rounded up to the next
    1-2-5 step: 0.025% gave the first floor, 0.05%."""
    target = 2.0 * float(np.median(null_medians))
    if target <= 0:
        raise ValueError("the null pairs measured no drift at all: nothing to declare")
    power = 10.0 ** math.floor(math.log10(target))
    return next(round(step * power, 12) for step in STEPS if step * power >= target * (1 - 1e-9))
