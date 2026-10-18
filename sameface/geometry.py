"""Fitting one image's points onto another's, and what a fit leaves behind.

A flat fit moves the whole after face as one sheet. A similarity can move,
scale and turn it; an affine can also stretch and slant it. Whatever is left
at each point after the best such sheet is the drift: the parts of the face
that sit somewhere else relative to the rest.

No torch: everything here is tested without a GPU.
"""
import math

import numpy as np

FITS = ("similarity", "affine")
MINIMUM_POINTS = {"similarity": 2, "affine": 3}


def umeyama(source, target):
    """Least-squares similarity (scale, rotation, translation) source -> target.

    Umeyama 1991. The reflection guard matters: without it a noisy or
    near-degenerate point set can fit a mirrored "rotation".
    """
    source, target = np.asarray(source, float), np.asarray(target, float)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    centred_source, centred_target = source - source_mean, target - target_mean

    u, singular, vt = np.linalg.svd(centred_source.T @ centred_target / len(source))
    correction = np.array([1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = (u @ np.diag(correction) @ vt).T
    scale = float((singular * correction).sum() * len(source) / (centred_source ** 2).sum())

    return scale, rotation, target_mean - scale * rotation @ source_mean


def fit(source, target, mode="similarity"):
    """The 2x3 that carries `source` onto `target` as closely as `mode` can."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    needed = MINIMUM_POINTS[mode]
    if len(source) < needed:
        raise ValueError(f"{mode} needs at least {needed} points, got {len(source)}")
    if mode == "similarity":
        scale, rotation, translation = umeyama(source, target)
        return np.column_stack([scale * rotation, translation])
    homogeneous = np.hstack([source, np.ones((len(source), 1))])
    return np.linalg.lstsq(homogeneous, target, rcond=None)[0].T


def apply(matrix, points):
    matrix = np.asarray(matrix, float)
    return np.asarray(points, float) @ matrix[:, :2].T + matrix[:, 2]


def point_set_size(points):
    """Twice the RMS radius about the centroid: what "% of face size" is of."""
    points = np.asarray(points, float)
    centre = points.mean(axis=0)
    return 2.0 * np.sqrt(((points - centre) ** 2).sum(axis=1).mean())


def residuals(fitted, target):
    """Each point's distance from fit to truth, as a percentage of size.

    Normalised by the target point set, so the number means "how far off,
    relative to how big the face is", and compares between a 512px image and
    a 4K one.
    """
    fitted, target = np.asarray(fitted, float), np.asarray(target, float)
    distances = np.hypot(*(fitted - target).T)
    return 100.0 * distances / point_set_size(target)


def describe(matrix, source, target):
    """A fitted 2x3 as scale, turn, stretch, shear and how far the face moved.

    The stretches are the lengths of the matrix's two columns - what happens
    to the after image's x and y axes - and the shear is how far the angle
    between them is from a right angle. The offset is source centre minus
    target centre: where the after face is, relative to the before face.
    """
    matrix = np.asarray(matrix, float)
    linear = matrix[:, :2]
    across, down = linear[:, 0], linear[:, 1]
    lean = math.degrees(math.atan2(down[1], down[0]) - math.atan2(across[1], across[0]))
    offset = np.asarray(source, float).mean(axis=0) - np.asarray(target, float).mean(axis=0)
    return {
        "scale": float(np.sqrt(abs(np.linalg.det(linear)))),
        "rotation_deg": float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))),
        "stretch_x": float(np.hypot(*across)),
        "stretch_y": float(np.hypot(*down)),
        # Wrapped into (-180, 180] before the right angle is taken off.
        "shear_deg": float((lean + 180.0) % 360.0 - 180.0 - 90.0),
        "offset_x": float(offset[0]),
        "offset_y": float(offset[1]),
    }


def cover(image_size, frame_size):
    """Scale and centre-crop offset for fitting an image over a frame.

    `max` rather than `min` is the difference between cover and contain: the
    frame is filled and the excess hangs off both sides equally.
    """
    image, frame = np.asarray(image_size, float), np.asarray(frame_size, float)
    scale = (frame / image).max()
    return scale, (image * scale - frame) / 2.0


def into_frame(points, image_size, frame_size):
    """Image pixels to frame pixels, under a cover fit."""
    scale, offset = cover(image_size, frame_size)
    return np.asarray(points, float) * scale - offset
