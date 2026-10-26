"""SameFace Simple: a before and an after in, the answer out.

The advanced nodes' own code at the pack's defaults, so Simple's figures are
Measure's for the same pair. What Simple decides by itself:

- **The fit**, whichever of similarity and affine leaves the smaller median
  on the edit - an average user should not have to know the two. The repair
  is reported after the same fit, so its figure compares with the edit's.
- **A repair**, unless switched off: at full strength, in the after's own
  framing, and measured against the before - only a fresh measurement on the
  new pixels says what the repair left.
- **The face**: the surest one on each side. Choosing one of several people
  is the advanced nodes' job (Face Mask into Measure).
"""
from dataclasses import dataclass

from .geometry import FITS
from .measure import Drift, measure, repair

# The views' arrows, ten times their length: at one, a tenth of a percent of
# drift is shorter than a pixel. The factor is printed on the picture.
EXAGGERATION = 10


@dataclass
class Result:
    drift: Drift                 # the edit measured against the before
    fit: str                     # the fit every figure and picture uses
    repaired: object = None      # the repaired after, uint8 RGB, or None
    repaired_drift: Drift = None  # the repair measured against the before

    @property
    def median(self):
        return self.drift.fits[self.fit].median

    @property
    def repaired_median(self):
        return None if self.repaired_drift is None else self.repaired_drift.fits[self.fit].median


def smaller_fit(drift):
    """The fit that leaves the smaller median."""
    return min(FITS, key=lambda mode: drift.fits[mode].median)


def simple(before, after, masker, flows, fix=True):
    """One pair, measured and, if `fix`, repaired and measured again."""
    drift = measure(before, after, masker, flows, ears=True)
    result = Result(drift=drift, fit=smaller_fit(drift))
    if fix:
        result.repaired, _ = repair(drift)
        # The before's face as already found: the same face, and one model
        # run fewer.
        result.repaired_drift = measure(before, result.repaired, masker, flows, ears=True,
                                        before_region=drift.before_region)
    return result


def report(result, label=None):
    """The figures in words: the drift, and what the repair left."""
    lines = [f"SameFace drift{f' - {label}' if label else ''}: "
             f"{result.median:.3f}% of face size"]
    if result.repaired_drift is not None:
        lines.append(f"after repair: {result.repaired_median:.3f}%")
    crowded = [f"{region.faces_found} in the {side}"
               for side, region in (("before", result.drift.before_region),
                                    ("after", result.drift.after_region))
               if region.faces_found > 1]
    if crowded:
        lines.append(f"faces found: {' and '.join(crowded)}; the surest measured. To "
                     "choose one, use SameFace Face Mask with SameFace Measure.")
    return "\n".join(lines)
