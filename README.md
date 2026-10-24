# SameFace

ComfyUI nodes that measure how far an image edit moved a face, and move it back.

An edit model asked to change the hair or the background often changes the face
a little too: an eye sits higher, the jaw is wider, the nose has moved. Each
change is small, but together they are why the result stops looking like the
same person. SameFace measures that drift and repairs it.

## Nodes

Measure runs the models and gives the figures. Every picture is drawn by a
node of its own from Measure's `drift`, and runs no models.

**SameFace Measure**: `before` + `after` →
- `report`: the figures in words
- `similarity_median`, `affine_median`: the headline figures, % of face size
- `before_mask`, `after_mask`: the face masks used
- `drift`: everything measured, for Repair and the nodes that draw it
- `points`: every measured point as JSON, one list per column (`pair`, `x`,
  `y`, `dx_`/`dy_`/`drift_` for each fit, `confidence`, `part`), for your own
  plots or analysis

**SameFace Shift Map**: `drift` + `fit` → `shift_map`

The before image with every measured point coloured by how far the fit left
it.

**SameFace Repair**: `drift` + `strength` → `repaired`, `covered`

**SameFace View**: `drift` → `before_view`, `after_view`

The before and the after in one frame, with the measured points drawn on: green
on the before, red on the after with an arrow to where the before has each
point, and a mesh joining the points, the same triangles on both sides, so a
part of the face that drifted shows as triangles changing shape. Feed both into
**Compare Images** for a slider, or into Save Image to keep them. The after can
be shown moved by the flat fit (the arrows are then the drift) or repaired. An
exaggeration setting stretches the arrows only and is printed on the picture.
Orange crosses on the before mark the gaps in the grid: points the flow could
not vouch for (see below).

**SameFace Drift Histogram**: `drift` → `histogram`

How many points drifted how far: bars for the similarity fit, an outline for
the affine fit, each median marked, and the noise floor dashed.

**SameFace Drift Zones**: `drift` + `fit`, `rows`, `cols` → `zones`

The before face cut into a grid (4 × 3 by default), each cell tinted by the
median drift of its points, in the shift map's colours, with the figure
printed. The cells are a grid over the face, not named features: a cell over
the eyes is not "the eyes".

Measure the repaired image again to see what is left. The warp puts its own
points where they belong by construction, so only a fresh measurement on the
new pixels can say whether it helped.

Batches: one before with any number of afters, or as many befores as afters.

## How it measures

1. **Face mask**: Grounding DINO finds the face and the ears from the words
   "face" and "ear"; SAM 2 outlines them. Ears in, neck out.
2. **Optical flow**: each side is cropped to its own face and resampled to one
   working size, so a resize or a reframing is not mistaken for drift. RAFT then
   follows every pixel of the before face into the after, forwards and
   backwards. RAFT gives no confidence of its own, so this is the check: a
   point that does not come back to where it started is left out. That
   happens where the edit rewrote texture or something covers the face. The
   rest, a few hundred matched points across the face, are measured.
3. **Flat fit**: a similarity (move, scale, turn) and an affine (also stretch
   and slant) are fitted through the points. What each fit leaves at a point is
   the drift there, as a percentage of face size.
   - Large under similarity but small under affine: the edit mostly stretched the face.
   - Large under both: features moved against each other.
4. **Repair**: the similarity fit, then a smooth warp: a thin-plate spline
   through the points, eased to nothing just beyond the face, so hair and
   background are left as the flat fit put them.

RAFT's noise floor is **0.05% of face size**: the median drift it reports on a
pure resample, where the true drift is zero. On the shift map, grey is within that
floor; the greens step up at 2, 5, 10 and 20 times it.

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

`example_workflows/` has a workflow that measures a pair, repairs it,
measures the repair, and shows the points on a before/after slider. Load your own before and after into the two Load Image
nodes.

## Tests

The geometry, the sampling and the warp are tested without a GPU:

```
python -m pytest
```

## Licence

Apache-2.0.
