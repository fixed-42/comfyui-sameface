"""Pictures and words for a measurement: the shift map, the mask, the report.

**The map's scale is fixed, never stretched to the image.** Grey within the
noise floor, then a green for each step between the edges chosen on the node
(by default 0.05, 0.1, 0.25, 0.5 and 1% of face size), so a colour is
the same shift on every pair drawn with the same steps, and the worst point
of a perfect pair stays grey. One hue, green, light to dark; nearly opaque
discs, because seen through, a photograph tints every step towards the same
colour.
"""
import cv2
import numpy as np

from .measure import dropped
from .smooth_warp import spacing_of

# Edges between the colours, % of face size: one step just above the noise
# floor, the rest over where real edits fall. Six greens is about as many as
# one hue keeps apart; seven ran together.
MAP_STEPS = (0.05, 0.1, 0.25, 0.5, 1.0)
MAX_STEPS = 8
# The ramp, light to dark; as many colours as the steps need are spread on it.
MAP_COLOURS = ((0xc8, 0xef, 0xbf), (0x8f, 0xd6, 0x7c), (0x45, 0xad, 0x42),
               (0x1c, 0x7d, 0x2a), (0x08, 0x3f, 0x14))
MAP_QUIET = (0xaa, 0xb0, 0xbd)
QUIET_ALPHA, STEP_ALPHA = 0.3, 0.95
OUTLINE = (255, 255, 255)


def parse_steps(text, floor):
    """The edges typed on a node, "0.1, 0.25, 0.5, 1", as floats - refused
    with the reason when they would draw a misleading map."""
    try:
        steps = tuple(float(part) for part in str(text).replace(";", ",").split(",")
                      if part.strip())
    except ValueError:
        raise ValueError(f"steps {text!r}: give numbers, % of face size, separated by "
                         "commas, such as 0.1, 0.25, 0.5, 1") from None
    if not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError(f"steps {text!r}: give 1 to {MAX_STEPS} of them")
    if any(low >= high for low, high in zip(steps, steps[1:])):
        raise ValueError(f"steps {text!r}: each must be larger than the one before")
    if steps[0] <= floor:
        raise ValueError(f"steps {text!r}: the first must be above the noise floor, "
                         f"{floor:g}% - below it everything is grey anyway")
    return steps


def map_colours(count):
    """`count` colours spread light to dark along `MAP_COLOURS`."""
    ramp = np.asarray(MAP_COLOURS, float)
    at = np.linspace(0, len(ramp) - 1, count) if count > 1 else np.array([len(ramp) // 2])
    low = np.minimum(np.floor(at).astype(int), len(ramp) - 2)
    share = (at - low)[:, None]
    return [tuple(int(round(v)) for v in colour)
            for colour in ramp[low] * (1 - share) + ramp[low + 1] * share]


def map_step(percent, floor, steps=MAP_STEPS):
    """Which colour a shift takes, or -1 where it is within the noise floor."""
    if percent <= floor:
        return -1
    return sum(percent > edge for edge in steps)


def map_legend(floor, steps=MAP_STEPS):
    def at(value):
        return f"{value:.3g}%"
    edges = (floor, *steps)
    return ([f"up to {at(floor)}"]
            + [f"{at(low)} - {at(high)}" for low, high in zip(edges, edges[1:])]
            + [f"over {at(steps[-1])}"])


def _blend(canvas, layer, mask, alpha):
    where = mask.astype(bool)
    canvas[where] = canvas[where] * (1 - alpha) + layer[where] * alpha


def shift_map(drift, mode, floor, steps=MAP_STEPS):
    """Each point as a disc on the before image, coloured by how much shift
    the `mode` fit left on it, in the colour of its step between `steps`.
    RGB uint8, the before image's size."""
    image = drift.before.astype(float)
    points = drift.before_points
    each = drift.fits[mode].residuals
    radius = max(1, int(round(spacing_of(points) / 2)))
    colours = map_colours(len(steps) + 1)
    taken = np.array([map_step(value, floor, steps) for value in each])

    for wanted, colour, alpha in ([(-1, MAP_QUIET, QUIET_ALPHA)]
                                  + [(step, colours[step], STEP_ALPHA)
                                     for step in range(len(colours))]):
        mask = np.zeros(image.shape[:2], np.uint8)
        for x, y in points[taken == wanted]:
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 1, -1, cv2.LINE_AA)
        layer = np.empty_like(image)
        layer[...] = colour
        _blend(image, layer, mask, alpha)

    # Lines and words are drawn in 8 bits: OpenCV 5 draws text on nothing else.
    image = np.clip(image, 0, 255).astype(np.uint8)
    scale = max(image.shape[:2]) / 1000.0
    thickness = max(1, int(round(2 * scale)))
    cv2.polylines(image, [np.round(drift.before_region.outline).astype(np.int32)],
                  True, OUTLINE, thickness, cv2.LINE_AA)
    _legend(image, mode, floor, steps, scale)
    return image


def _legend(image, mode, floor, steps, scale):
    """The steps' words in a box at the bottom left."""
    font, size = cv2.FONT_HERSHEY_SIMPLEX, 0.55 * max(scale, 0.5)
    weight = max(1, int(round(size * 2)))
    line = int(round(26 * max(scale, 0.5)))
    swatch = int(round(line * 0.7))
    rows = [(None, f"shift left by the {mode} fit")] + list(
        zip([MAP_QUIET, *map_colours(len(steps) + 1)], map_legend(floor, steps)))
    width = max(cv2.getTextSize(text, font, size, weight)[0][0] for _, text in rows)
    width += swatch + 3 * line // 2
    height = line * len(rows) + line // 2
    top = image.shape[0] - height - line // 2
    left = line // 2
    box = image[top:top + height, left:left + width]
    box[...] = (box * 0.35).astype(np.uint8)
    for row, (colour, text) in enumerate(rows):
        y = top + line * (row + 1)
        x = left + line // 2
        if colour is not None:
            cv2.rectangle(image, (x, y - swatch), (x + swatch, y), colour, -1)
            x += swatch + line // 2
        cv2.putText(image, text, (x, y - 2), font, size, (255, 255, 255), weight, cv2.LINE_AA)


def report(drift, floor, label=None):
    """The figures in words, one block per pair."""
    gaps = len(dropped(drift.record, drift.ears))
    lines = [f"SameFace drift{f' - {label}' if label else ''}",
             f"RAFT{', two passes' if drift.record.passes > 1 else ''}, "
             f"{len(drift.before_points)} points on the face"
             f"{' and ears' if drift.ears and not drift.before_region.supplied else ''}"
             f"; noise floor {floor}% of face size"]
    if gaps:
        lines.append(f"  {gaps} more dropped: the flow did not come back to where it started")
    for mode, fit in drift.fits.items():
        shape = fit.transform
        lines.append(
            f"  {mode:10} median {fit.median:.3f}%  p90 {fit.p90:.3f}%  "
            f"highest {fit.highest:.2f}%  above noise {100 * fit.above_noise:.0f}%")
        lines.append(
            f"  {'':10} scale {shape['scale']:.4f}  turn {shape['rotation_deg']:+.2f} deg"
            + (f"  stretch x {shape['stretch_x']:.4f} y {shape['stretch_y']:.4f}"
               f"  shear {shape['shear_deg']:+.2f} deg" if mode == "affine" else ""))
    for side, region in (("before", drift.before_region), ("after", drift.after_region)):
        lines.extend(_mask_lines(side, region))
    return "\n".join(lines)


def _mask_lines(side, region):
    """Where one side's face mask came from, and what to watch for in it."""
    if region.supplied:
        lines = [f"  mask {side}: supplied - all of it measured as face, the ears switch "
                 "has no effect"]
        if region.pieces_dropped:
            lines.append(f"    {region.pieces_dropped} smaller piece(s) of it left out, "
                         "the largest measured")
        if region.box_shaped:
            lines.append(f"    the {side} mask is box-shaped: hair and background inside it "
                         "are measured as face")
        return lines
    lines = [f"  mask {side}: face {region.face_score:.2f}, ears {region.ears_joined} joined "
             f"of {region.ears_found} found"]
    if region.faces_found > 1:
        lines.append(f"    {region.faces_found} faces found and the surest measured: if the "
                     "two sides could be different people, choose one with SameFace Face "
                     "Mask on each side")
    return lines
