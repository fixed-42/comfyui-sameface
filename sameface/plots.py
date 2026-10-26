"""One pair's points as data and as two pictures: a histogram of how far
they drifted, and the face cut into zones with each zone's median drift.

**The zones are a grid, not features.** They split the box around the
measured points into equal rows and columns; a cell over the eyes is not "the
eyes". Naming features needs landmarks, which RAFT does not give.

The zones use the shift map's fixed colour scale (`drawing.map_step`), so a
colour means the same drift on both pictures and on every pair.
"""
import cv2
import numpy as np

from .drawing import MAP_COLOURS, MAP_QUIET, MAP_STEPS, OUTLINE, map_colours, map_step
from .geometry import FITS, apply
from .measure import chosen_parts

BACKGROUND = (255, 255, 255)
INK = (40, 40, 40)
FAINT = (215, 215, 215)
FLOOR_LINE = (120, 120, 120)
SERIES = {"similarity": MAP_COLOURS[2], "affine": (0x5b, 0x6b, 0xc0)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
# The zones picture's height, whatever the face's size in the photo.
ZONES_HEIGHT = 1000
# A cell with fewer points than this is left empty: its median would be noise.
MIN_POINTS = 5
CELL_ALPHA = 0.55


def point_table(drifts):
    """Every measured point of every pair, one list per column, ready for JSON.

    `x`, `y` are where the point is on the before image, in its pixels. Each
    fit's `dx`, `dy` run from the before point to where the fit put its match
    in the after, in the same pixels; `drift` is their length as a percentage
    of face size, as everywhere else.
    """
    table = {name: [] for name in ("pair", "x", "y", "dx_similarity", "dy_similarity",
                                   "drift_similarity", "dx_affine", "dy_affine",
                                   "drift_affine", "confidence", "part")}
    for number, drift in enumerate(drifts):
        count = len(drift.before_points)
        table["pair"] += [number] * count
        table["x"] += drift.before_points[:, 0].tolist()
        table["y"] += drift.before_points[:, 1].tolist()
        for mode in FITS:
            fit = drift.fits[mode]
            offsets = apply(fit.matrix, drift.after_points) - drift.before_points
            table[f"dx_{mode}"] += offsets[:, 0].tolist()
            table[f"dy_{mode}"] += offsets[:, 1].tolist()
            table[f"drift_{mode}"] += np.asarray(fit.residuals, float).tolist()
        table["confidence"] += np.asarray(drift.confidence, float).tolist()
        table["part"] += chosen_parts(drift.record, drift.ears)
    return table


def zone_medians(points, drift, rows, cols, min_points=MIN_POINTS):
    """The median drift in each cell of a `rows` x `cols` grid over the
    points' bounding box; NaN where a cell holds fewer than `min_points`."""
    points, drift = np.asarray(points, float), np.asarray(drift, float)
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    col = np.clip(((points[:, 0] - low[0]) / span[0] * cols).astype(int), 0, cols - 1)
    row = np.clip(((points[:, 1] - low[1]) / span[1] * rows).astype(int), 0, rows - 1)
    medians = np.full((rows, cols), np.nan)
    for r in range(rows):
        for c in range(cols):
            here = drift[(row == r) & (col == c)]
            if len(here) >= min_points:
                medians[r, c] = np.median(here)
    return medians


def same_size(images):
    """Pad each image with white at the right and bottom to the largest size
    among them, so a batch of pictures of different pairs stacks."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        canvas = np.empty((height, width, 3), np.uint8)
        canvas[...] = BACKGROUND
        canvas[:image.shape[0], :image.shape[1]] = image
        padded.append(canvas)
    return padded


def _text(image, text, origin, size, colour, weight, anchor="left"):
    width = cv2.getTextSize(text, FONT, size, weight)[0][0]
    x = {"left": origin[0], "centre": origin[0] - width // 2, "right": origin[0] - width}[anchor]
    cv2.putText(image, text, (int(x), int(origin[1])), FONT, size, colour, weight, cv2.LINE_AA)


def _tick_step(top, most=6):
    """1, 2 or 5 times a power of ten, putting at most `most` ticks under `top`."""
    raw = top / most
    power = 10 ** np.floor(np.log10(raw))
    return next(step * power for step in (1, 2, 5, 10) if step * power >= raw)


def drift_histogram(drift, floor, size=(1200, 600), bins=40):
    """How many points drifted how far, after each fit, with the noise floor
    and each fit's median marked. RGB uint8; `size` is (width, height)."""
    fits = drift.fits
    return _histogram(
        [(fits["similarity"].residuals, fits["similarity"].median, SERIES["similarity"],
          f"similarity fit: bars, median {fits['similarity'].median:.3f}%"),
         (fits["affine"].residuals, fits["affine"].median, SERIES["affine"],
          f"affine fit: outline, median {fits['affine'].median:.3f}%")],
        floor, f"drift per point, % of face size ({len(fits['similarity'].residuals)} points)",
        size, bins)


def repair_histogram(drift, mode, floor, repaired=None, size=(1200, 600), bins=40):
    """The edit's drift after the `mode` fit as bars and, given `repaired`,
    the repair's after the same fit as an outline over them: the one further
    left drifted less."""
    edit = drift.fits[mode]
    series = [(edit.residuals, edit.median, SERIES["similarity"],
               f"the edit: bars, median {edit.median:.3f}%")]
    if repaired is not None:
        fixed = repaired.fits[mode]
        series.append((fixed.residuals, fixed.median, SERIES["affine"],
                       f"after repair: outline, median {fixed.median:.3f}%"))
    return _histogram(series, floor, "drift per point, % of face size", size, bins)


def _histogram(series, floor, title, size, bins):
    """The first of `series` as filled bars, a second as an outline over them,
    so both stay readable where they overlap; each `(residuals, median,
    colour, label)`, its median marked in its colour."""
    width, height = size
    image = np.empty((height, width, 3), np.uint8)
    image[...] = BACKGROUND
    scale = height / 600.0
    small, normal = 0.5 * scale, 0.6 * scale
    weight = max(1, int(round(1.5 * scale)))
    left, right = int(0.08 * width), int(0.03 * width)
    top, bottom = int(0.17 * height), int(0.14 * height)
    plot_w, plot_h = width - left - right, height - top - bottom
    base = top + plot_h

    values = [np.asarray(residuals, float) for residuals, *_ in series]
    reach = max(max(np.percentile(each, 99) for each in values), 3 * floor) * 1.05
    edges = np.linspace(0.0, reach, bins + 1)
    counts = [np.histogram(np.clip(each, 0, reach), edges)[0] for each in values]
    tallest = max(max(each.max() for each in counts), 1) * 1.1

    def x_of(value):
        return int(round(left + plot_w * min(value, reach) / reach))

    def y_of(count):
        return int(round(base - plot_h * count / tallest))

    for tick in np.arange(0.0, reach, _tick_step(reach)):
        x = x_of(tick)
        cv2.line(image, (x, top), (x, base), FAINT, 1)
        _text(image, f"{tick:.3g}%", (x, base + 22 * scale), small, INK, weight, "centre")
    for tick in np.arange(0, tallest, _tick_step(tallest)):
        _text(image, f"{int(tick)}", (left - 8, y_of(tick) + 5 * scale), small, INK, weight, "right")

    for i, count in enumerate(counts[0]):
        if count:
            cv2.rectangle(image, (x_of(edges[i]) + 1, y_of(count)), (x_of(edges[i + 1]) - 1, base),
                          series[0][2], -1)
    if len(series) > 1:
        outline = []
        for i, count in enumerate(counts[1]):
            outline += [(x_of(edges[i]), y_of(count)), (x_of(edges[i + 1]), y_of(count))]
        outline = [(x_of(0), base)] + outline + [(x_of(edges[-1]), base)]
        cv2.polylines(image, [np.array(outline, np.int32)], False, series[1][2], weight + 1,
                      cv2.LINE_AA)

    for _, median, colour, _ in series:
        x = x_of(median)
        cv2.line(image, (x, top), (x, base), colour, weight)
    dash = max(6, int(10 * scale))
    x = x_of(floor)
    for y in range(top, base, 2 * dash):
        cv2.line(image, (x, y), (x, min(y + dash, base)), FLOOR_LINE, weight)
    cv2.line(image, (left, base), (left + plot_w, base), INK, weight)

    _text(image, title, (left, int(0.06 * height)), normal, INK, weight)
    line = int(0.06 * height) + int(28 * scale)
    for (_, _, colour, label), x in zip(series, (left, left + plot_w // 2 - int(60 * scale))):
        _text(image, label, (x, line), small, colour, weight)
    _text(image, f"dashed: noise floor {floor:g}%", (left + plot_w, line), small,
          FLOOR_LINE, weight, "right")
    _text(image, "points", (left - 8, top - int(10 * scale)), small, INK, weight, "right")
    return image


def drift_zones(drift, mode, floor, rows=4, cols=3, min_points=MIN_POINTS, steps=MAP_STEPS):
    """The before face, cut into a `rows` x `cols` grid over its measured
    points, each cell tinted by its median drift after the `mode` fit, in the
    shift map's colour for its step, and the figure printed in it. RGB uint8,
    `ZONES_HEIGHT` tall."""
    colours = map_colours(len(steps) + 1)
    points = drift.before_points
    medians = zone_medians(points, drift.fits[mode].residuals, rows, cols, min_points)
    low, high = points.min(axis=0), points.max(axis=0)

    height, width = drift.before.shape[:2]
    pad = 0.12 * (high - low).max()
    x0, y0 = (max(int(v - pad), 0) for v in low)
    x1, y1 = int(min(high[0] + pad, width)), int(min(high[1] + pad, height))
    band = int(ZONES_HEIGHT * 0.05)
    zoom = (ZONES_HEIGHT - band) / (y1 - y0)
    image = cv2.resize(drift.before[y0:y1, x0:x1],
                       (max(1, round((x1 - x0) * zoom)), ZONES_HEIGHT - band),
                       interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_CUBIC)

    def at(x, y):
        return int(round((x - x0) * zoom)), int(round((y - y0) * zoom))

    layer = image.copy()
    cell_w, cell_h = (high - low) / (cols, rows)
    corners = {}
    for r in range(rows):
        for c in range(cols):
            corner = at(low[0] + c * cell_w, low[1] + r * cell_h)
            far = at(low[0] + (c + 1) * cell_w, low[1] + (r + 1) * cell_h)
            corners[r, c] = corner, far
            if not np.isnan(medians[r, c]):
                step = map_step(medians[r, c], floor, steps)
                cv2.rectangle(layer, corner, far, MAP_QUIET if step < 0 else colours[step], -1)
    image = cv2.addWeighted(layer, CELL_ALPHA, image, 1 - CELL_ALPHA, 0)

    outline = np.array([at(x, y) for x, y in drift.before_region.outline], np.int32)
    cv2.polylines(image, [outline], True, OUTLINE, 2, cv2.LINE_AA)
    size = 0.9 * min(cell_w * zoom, cell_h * zoom) / 160
    weight = max(1, int(round(size * 2.5)))
    for (r, c), (corner, far) in corners.items():
        cv2.rectangle(image, corner, far, OUTLINE, 1)
        if np.isnan(medians[r, c]):
            continue
        text = f"{medians[r, c]:.2f}%"
        (w, h), _ = cv2.getTextSize(text, FONT, size, weight)
        x, y = (corner[0] + far[0] - w) // 2, (corner[1] + far[1] + h) // 2
        cv2.putText(image, text, (x, y), FONT, size, (0, 0, 0), weight + 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), FONT, size, (255, 255, 255), weight, cv2.LINE_AA)

    # The caption shrinks to the picture's width: a narrow face makes a
    # narrow picture.
    caption = f"median drift per zone, {mode} fit; colours as on the shift map"
    framed = np.empty((ZONES_HEIGHT, image.shape[1], 3), np.uint8)
    framed[...] = BACKGROUND
    framed[band:] = image
    size = min(0.9, 0.9 * (image.shape[1] - 20) / cv2.getTextSize(caption, FONT, 0.9, 2)[0][0])
    _text(framed, caption, (10, int(band * 0.72)), size, INK, 2)
    return framed
