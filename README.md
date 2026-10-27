# SameFace

**Measure and repair face drift in ComfyUI**: the small shifts an edit makes
in a face's shape, too small for the eye to catch.

Restore an old photo, colourise it, or ask an edit model for new hair or a
new background, and the face often comes back slightly different: an eye a
little higher, the jaw a little wider, the nose moved. People are very good at
recognising who a face belongs to - so good that we still see the same person
after the face has been altered. So the changes slip past us: today's edit
models make each one too small to notice, and together they still move the
face away from the person it was.

**SameFace puts a number on that drift and shows where it is**, in figures and
plots. Then it attempts the last mile: it moves the edited face's features
back to where the original has them, and measures the result again. One node,
**SameFace Simple**, does all of this; a set of advanced nodes does each step
with every setting.

![Lincoln's 1863 portrait with its measured points, the restored and colourised edit with arrows to where each point should be, and the drift of the edit against the repair](docs/images/hero.png)

*Lincoln, 1863, restored and colourised by an edit model. The edit moved the
face by 0.073% of its size. After SameFace's repair, 0.009%: below the noise
floor, so what is left cannot be told from the measurement's own noise.*

**Demo video** 

https://github.com/user-attachments/assets/37e553e2-7d3a-49a6-8891-a112d9e83152

## What it is for

- **Old-photo restoration and colourisation.** Check that the restored face is
  still the same person's, see where it changed, and put it back.
- **Face edits.** New hair, background, lighting or clothes, and the same face.
- **Comparing edit models and workflows.** One figure per pair, on the same
  scale for every model, down to drift far too small for the eye - with the
  pictures to show where it is.
- **Character LoRA datasets.** A character LoRA learns whatever face its
  training images show. If upscaling or enhancing an image drifted the face,
  the LoRA learns a slightly different person: measure each processed image
  against its source before it goes into the dataset.

## Quick start

1. Install (below) and restart ComfyUI.
2. Open `example_workflows/sameface_simple.json`, and copy the two Lincoln
   pictures from `example_workflows/input/` into `ComfyUI/input/`.
3. Run. The first Simple measures and repairs; the second measures the
   repair, for a slider of what is left. Then load your own original into
   **Before** and the edited picture into **After**.

The sliders are ComfyUI's Compare Images, which shows its slider with
**Modern Node Design (Nodes 2.0)** turned on (Settings → Comfy → Nodes 2.0);
rgthree's Image Comparer works in its place. Each example has a note saying
so.

**SameFace Simple** shows its figures on the node itself - the edit's drift,
and what the repair left - so the node alone tells you how far the face
moved, with nothing wired to its outputs. The outputs give the pictures, and
the numbers to wire on:

| Output | |
|---|---|
| `before_view`, `after_view` | into Compare Images for a slider: the measured points, with arrows (ten times their length) to where each point should be |
| `repaired` | the edit with its face moved back; everything else, and the framing, as the edit left them |
| `shift_map` | where on the face it drifted: grey within the noise, darker green for more |
| `histogram` | how far the points drifted, the edit as bars and the repair as an outline: the one further left drifted less |
| `median`, `repaired_median` | the figures, % of face size |

It measures the surest face on each side. Turn `repair` off to only measure.

![The before with its points, the edit with arrows, the shift map and the repaired image](docs/images/simple_outputs.jpg)

*An edit that put a fan in her hand moved the face by 0.032%; repaired,
0.013%. Left to right: `before_view`, `after_view`, `shift_map`, `repaired`.*

## The advanced nodes

Every step on its own node with all its settings, for choosing one face among
several, measuring the whole head, or drawing your own plots. All are in
**[The nodes](docs/nodes.md)**.

`example_workflows/sameface_advanced.json`, with its two pictures in
`example_workflows/input/`, is a family photo edited onto a museum backdrop: Face Mask chooses the mother's head on each side, Measure
measures it, Repair moves it back, and a second Measure measures the repair -
0.211% to 0.019% after the affine fit - with a view, a shift map and a
histogram of each.

| Node | |
|---|---|
| SameFace Measure | runs the models and gives the figures and `drift`, which the others read |
| SameFace Face Mask | one person's face or head, for Measure's mask inputs |
| SameFace Repair | moves the edit's face back, keeping its framing or lining it up with the before |
| SameFace View | the before and after with the points and a mesh, for a slider |
| SameFace Shift Map | the before with each point coloured by its drift |
| SameFace Drift Histogram | how far the points drifted, after each fit |
| SameFace Drift Zones | the face as a grid, each cell's median drift |

## How it works

1. **The face.** Grounding DINO finds it from the word "face", with the ears;
   SAM 2 outlines it.
2. **Every point of it.** RAFT optical flow follows each pixel of the before
   face into the edit and back. A point that does not return to where it
   started is dropped, so texture the edit rewrote is not mistaken for
   movement. A second pass, with the two faces lined up first, keeps the
   framing out of the figures.
3. **The drift.** The move, scale and turn of the whole face are fitted and
   taken out; what is left at each point is drift, in % of the face's size.
4. **The repair.** A smooth warp through the measured points, eased to
   nothing beyond the face, moves each feature back. SameFace Simple then
   measures the repaired image again - only a fresh measurement says what the
   repair left; with the advanced nodes, wire a second Measure.

The **noise floor is 0.01% of face size**: the drift the measurement gives a
face against a resized, recompressed copy of itself, which has no true drift.
It is re-measured by a tool in this repo, on your own faces if you like. The
method, the fits and the limits are in
**[How it measures](docs/how-it-measures.md)**.

## Open weights

No training, no API, nothing leaves the machine. The models download on first
use, about 1.8 GB, pinned to fixed revisions:

| Model | Used for | Licence |
|---|---|---|
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) (base) | finding the face, ears and head from words | Apache-2.0 |
| [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) (Hiera large) | outlining them | Apache-2.0 |
| [RAFT](https://pytorch.org/vision/stable/models/raft.html) (large, torchvision) | the optical flow | BSD-3-Clause |

## Why not face landmarks (MediaPipe, FAN)

Landmark detectors were studied first, before SameFace was built, and they are
too coarse for drift this small. On null pairs - a face against a resized,
recompressed copy of itself, so the true drift is zero - MediaPipe's
landmarks moved 0.19% of face size and FAN's 0.57%, against RAFT's 0.05% on
the same pairs (0.01% now, with the second pass). The edits SameFace measures
drift by 0.1-0.3%, so landmark noise is as large as the drift itself.
Landmarks are also sparse - FAN has 68 named points, where RAFT matches
several hundred across the face - and the detectors were weakest on profile
and head-down views. FAN's weights also come with unclear licence terms.

## Install

Clone into `ComfyUI/custom_nodes/` and install the requirements into
ComfyUI's environment:

```
cd ComfyUI/custom_nodes
git clone https://github.com/fixed-42/comfyui-sameface
pip install -r comfyui-sameface/requirements.txt
```

Restart ComfyUI; the nodes are under **SameFace**.

## Tests

Everything but the models themselves is tested without a GPU - the masks,
the sampling, the fits, the warp, the pictures, Simple and the noise floor's
rule:

```
python -m pytest
```

`tests/test_models.py` checks which detections the face masker counts, with
the models stood in for; it needs torch, so it is skipped without it. Run it
with ComfyUI's own Python to include it.

## Licence

Apache-2.0.

The pictures: Lincoln by Alexander Gardner, 1863 (Library of Congress,
LC-DIG-ppmsca-19301), and the family from the Farm Security Administration/
Office of War Information collection (Library of Congress), both in the public
domain; the edits of them, and the portrait with the fan, were made for this
project with local models.
