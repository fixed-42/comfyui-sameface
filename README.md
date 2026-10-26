# SameFace

ComfyUI nodes that measure how far an image edit moved a face, and move it back.

An edit model asked to change the hair or the background often changes the face
a little too: an eye sits higher, the jaw is wider, the nose has moved. Each
change is small, but together they are why the result stops looking like the
same person. SameFace measures that drift and repairs it.

## Nodes

Measure runs the models and gives the figures. Every picture is drawn by a
node of its own from Measure's `drift`, and runs no models.

**SameFace Measure**: `before` + `after` (+ `before_mask`, `after_mask`,
optional) →
- `report`: the figures in words
- `similarity_median`, `affine_median`: the headline figures, % of face size
- `before_mask`, `after_mask`: the face masks used
- `drift`: everything measured, for Repair and the nodes that draw it
- `points`: every measured point as JSON, one list per column (`pair`, `x`,
  `y`, `dx_`/`dy_`/`drift_` for each fit, `confidence`, `part`), for your own
  plots or analysis

Without a mask, Measure finds the face itself, and when there are several
it measures the surest on each side; the report says when that happened.
A wired mask is used instead, on that side only: everything inside it is
measured as face, so `ears` has no effect there. One mask serves every image,
or give one per image; a mask must be its image's size. A box-shaped mask (a
detector's bounding box) is measured, and the report warns that hair and
background inside it count as face.

**SameFace Face Mask**: `image` + `region`, `order`, `index`, `ears`, `confidence`, `gap`, `ear_confidence` → `mask`, `faces_found`

One person's face when there are several: the faces are counted `left to
right` (by the middle of each face) or `surest` first, and the one at `index`,
from 0, is masked. Wire one into each of Measure's `before_mask` and
`after_mask`, choosing the same person on both sides. Left to right is the
default because it stays put when two faces score nearly the same, which
`surest` does not; if the edit moved people around, the index can differ
between the sides. `faces_found` says how many there were.

`confidence` (default 0.5) is the detector score a face or head needs to be
counted. On one family photo the people score 0.58 to 0.77, a face on
a poster behind them 0.34 to 0.54, and nothing else above 0.13. Raise it to
leave out a face you don't want counted, lower it for one that is missed.
Mind that the same face can score differently on the two sides - the poster
scores 0.43 in the source and 0.54 in the edit, so at 0.5 it is counted in one
and not the other, and every index after it shifts. Compare `faces_found` on
both sides. Measure finding its own face (no mask wired) keeps 0.25.

Only what is counted - the faces, or the heads - uses `confidence`. The parts
of the person chosen are found by where they sit: the face inside a head at
0.25, and the ears beside the face at `ear_confidence` (default 0.25). Ears
score low, 0.22 to 0.28 on that photo, so lower it a little if one is
missed; a false ear is taken only if it sits beside the face, is much smaller
than it and joins it across the gap.

`gap` (default 4, in % of the face's diagonal) is how far apart the face and
its ears, or the face and its hair, may be and still be joined into one mask.
Widen it when a head mask stops at the hairline. Hair further off than the
gap is an error rather than a head mask that is only a face. Closing a gap
also fills notches in the outline narrower than it, so keep it no wider than
it needs to be.

`region` `head` masks the whole head, hair included, so the head's shape is
measured as well as the face. Use it only when the edit left the hair alone:
RAFT does not know hair from face, so hair the edit changed reads as drift.
Hair is also low-texture and measures noisier than skin, and the noise floor
was measured on faces, so a head's figures are not comparable with a face's.
Repair from a face measurement: a head measurement would move the hair as
well, and the hair edge with it. To see the head's figures, add a second
Measure with head masks for reading only.

**SameFace Shift Map**: `drift` + `fit`, `steps` → `shift_map`

The before image with every measured point coloured by how far the fit left
it: grey within the noise floor, then a darker green at each of `steps`, in %
of face size. The default, `0.05, 0.1, 0.25, 0.5, 1`, gives six greens, which
suits most edits; give 1 to 8 steps of your own, smallest first.

**SameFace Repair**: `drift` + `strength`, `framing` → `repaired`, `covered`

Moves the after's features back to where the before has them. With `framing`
`after` (the default) the repaired image keeps the edit's framing and size and
only the face changes. With `before`, the whole after is lined up with the
before, enlarged so its detail is kept; where it does not reach is black, and
`covered` says where that is.

**SameFace View**: `drift` → `before_view`, `after_view`

The before and the after in one frame, with the measured points drawn on: green
on the before, red on the after with an arrow to where the before has each
point, and a mesh joining the points, the same triangles on both sides, so a
part of the face that drifted shows as triangles changing shape. Feed both into
**Compare Images** for a slider, or into Save Image to keep them. The after is
moved by the fit you choose, so the arrows are the drift it leaves. An
exaggeration setting stretches the arrows only and is printed on the picture.
Orange crosses on the before mark the gaps in the grid: points the flow could
not vouch for (see below).

**SameFace Drift Histogram**: `drift` → `histogram`

How many points drifted how far: bars for the similarity fit, an outline for
the affine fit, each median marked, and the noise floor dashed.

**SameFace Drift Zones**: `drift` + `fit`, `rows`, `cols`, `steps` → `zones`

The before face cut into a grid (4 × 3 by default), each cell tinted by the
median drift of its points, in the shift map's colours, with the figure
printed. The cells are a grid over the face, not named features: a cell over
the eyes is not "the eyes".

Measure the repaired image against the before to see what is left, and View
that measurement for a before/repaired slider. The warp puts its own points
where they belong by construction, so only a fresh measurement on the new
pixels can say whether it helped.

Batches: one before with any number of afters, or as many befores as afters.

## How it measures

1. **Face mask**: Grounding DINO finds the face and the ears from the words
   "face" and "ear"; SAM 2 outlines them. Ears in, neck out. For a head, SAM 2
   outlines the hair in a "head" box and the face inside it is joined on. Or
   the mask you wire in.
2. **Optical flow**: each side is cropped to its own face and resampled to one
   working size, so a resize or a reframing is not mistaken for drift. RAFT then
   follows every pixel of the before face into the after, forwards and
   backwards. RAFT gives no confidence of its own, so this is the check: a
   point that does not come back to where it started is left out. That
   happens where the edit rewrote texture or something covers the face. The
   rest, a few hundred matched points across the face, are measured.
   Then a second pass: the after is cut again where a similarity fit through
   those points puts the before's crop, turned and scaled with it, and RAFT
   runs once more on the two faces lined up. With the first pass alone, the
   turn and scale left between two box crops read partly as drift, so the
   same edit measured lower once it had been moved onto the before.
3. **Flat fit**: a similarity (move, scale, turn) and an affine (also stretch
   and slant) are fitted through the points. What each fit leaves at a point is
   the drift there, as a percentage of face size.
   - Large under similarity but small under affine: the edit mostly stretched the face.
   - Large under both: features moved against each other.
4. **Repair**: a smooth warp, a thin-plate spline through the points, eased
   to nothing just beyond the face, so hair and background are left as they
   were. In the after's framing, the before's points are first carried into
   the after by the after's own pose; in the before's, the after is first
   moved by the similarity fit.

The noise floor is **0.01% of face size**: below it, a shift cannot be told
from the measurement's own noise. It comes from null pairs, each face resampled
to 512 and 1414 pixels and saved as a JPEG, so the true drift is zero: twice
their median drift (0.0033% over 25 faces), rounded up to a 1-2-5 step. On the
shift map, grey is within that floor. To measure it on your own faces, with
ComfyUI's Python, from this folder:

    python tools/calibrate_noise_floor.py <folder of face images>

## Install

Clone into `ComfyUI/custom_nodes/` and install the requirements into
ComfyUI's environment:

```
cd ComfyUI/custom_nodes
git clone https://github.com/fixed-42/comfyui-sameface
pip install -r comfyui-sameface/requirements.txt
```

The models download on first use (about 1.8 GB) into the Hugging Face and
torch caches. All are open weights: Grounding DINO and SAM 2.1 under
Apache-2.0, and RAFT (torchvision) under BSD-3-Clause.

`example_workflows/` has two workflows, a restored 720p frame and a
hair-and-background edit. Each measures a pair, repairs it, measures the
repair, and shows the points on a before/after slider. Load your own before
and after into the two Load Image nodes.

## Tests

Everything but the models themselves is tested without a GPU - the masks,
the sampling, the fits, the warp, the pictures and the noise floor's rule:

```
python -m pytest
```

`tests/test_models.py` checks which detections the face masker counts, with
the models stood in for; it needs torch, so it is skipped without it. Run it
with ComfyUI's own Python to include it.

## Licence

Apache-2.0.
