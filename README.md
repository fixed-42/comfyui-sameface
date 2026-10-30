# SameFace

**ComfyUI nodes that measure where an image edit moved the parts of a face,
and move them back.** Open weights, runs locally.

**Can you spot the difference?**

![Lincoln's 1863 portrait and an edit that colourised and restored it, side by side at the same scale](docs/images/spot_the_difference.jpg)

*Lincoln, 1863 (left), and an edit that colourised and restored it (right),
lined up at the same scale. Hint: the head on the right is 3.7% wider for its
height.*

SameFace follows every point of a face from the original into the edit with
optical flow (RAFT, inside a Grounded SAM 2 face mask), takes out the move,
scale and turn of the whole face, and reports what is left - the drift - in %
of face size, with plots of where it is. On the pair above it measured a
drift of 0.83% of the head's size, and the 3.7% comes from the same
measurement: to match the original, the edit's head had to be stretched 3.7%
more in height than in width.

**Ok, so what happened**

Restore an old photo, colourise it, or ask an edit model for new hair or a
new background, and the face often comes back slightly different: an eye a
little higher, the jaw a little wider, the nose moved. You may sense that
something changed, but not where or by how much - and because we recognise
who a face belongs to so well, it still looks like the same person, so the
change is easy to let pass. SameFace says where the face moved and how far,
on one scale for every edit, and moves it back. It measures geometry - where
the parts of the face went - not likeness.

**Demo video**

https://github.com/user-attachments/assets/69bfca1b-3fb7-49b5-9145-54396f8a57ab

## What it is for

SameFace also moves the face back. On the Lincoln pair, its repair - a
thin-plate spline warp through the measured points - took the whole head's
drift from 0.83% to a measured 0.010%. A fresh measurement reads what a repair
leaves too low, though: on a known warp of the same size the truth was 0.033%,
so expect about 0.03% here, with about 96% of the drift gone. The rest of the
picture stays as the edit made it.

- **Old-photo restoration and colourisation.** See whether the restored face
  kept its shape, where it changed, and put it back.
- **Face edits.** New hair, background, lighting or clothes, and the same face.
- **Comparing edit models and workflows.** One figure per pair, on the same
  scale for every model, with pictures of where the drift is.
- **Character LoRA datasets.** A character LoRA learns whatever face its
  training images show. If upscaling or enhancing an image drifted the face,
  the LoRA learns a slightly different person: measure each processed image
  against its source before it goes into the dataset.

## Why optical flow, not face landmarks

SameFace comes out of a study of how to measure this at all. Aligning two
faces on a few points lines them up but cannot measure them: an affine fit
through three points leaves nothing over, for any two faces. So the study
turned to face landmark detectors - MediaPipe's 478-point mesh and FAN's 68
points - and found them too coarse for drift this small. On null pairs, a
face against a resized copy of itself, where the true drift is zero, their
points moved about as far as most edits drift:

| Null pairs: 25 faces, each resized to 512 and 1414 px | Median drift, % of face size |
|---|---|
| MediaPipe Face Mesh | 0.19% |
| FAN 2D / 3D | 0.64% / 0.56% |
| RAFT optical flow, one pass | 0.025% |
| RAFT, SameFace's two passes | 0.003% |

The edits in this README drift 0.03-0.8%. Landmarks could see the largest,
but not the small ones, and neither could check what a repair leaves: about
0.03% on Lincoln, a sixth of MediaPipe's noise. A second test, against 20,434
human-marked points on 284 faces from public landmark datasets, showed the
same from another side: the detectors' points moved under changes that move
nothing - blur, JPEG, brightness - and MediaPipe placed the outer eye corners
inside where people mark them, on three faces in four.

Optical flow has no face model to be confidently wrong, and gives several
hundred points per face. The parts are known - RAFT, a forward-backward
check, similarity and affine fits, a thin-plate spline. What SameFace adds is
putting them to this question: a dense measurement of the drift an edit
leaves in a face, a second pass that keeps the framing out of the figure, a
noise floor anyone can re-measure, and a repair that is measured again rather
than trusted.

The **noise floor is 0.01% of face size**: twice the median drift the
measurement gives a face against a resized, recompressed copy of itself,
which has no true drift. A tool in this repo re-measures it, on your own
faces if you like. It says how far the measurement repeats, not how close it
comes to the truth.

To test that, warps of known size were put through the real models. Measure
read 99% of a 0.88% drift, 92% of 0.24%, 71% of 0.07% and 42% of 0.024%. What a
repair really left was then worked out two ways with no RAFT in them, which
agree. It is all in **[Checking against known
warps](docs/known-warp-check.md)**, with a tool to run it yourself. The method,
the figures, their caveats and the study are in
**[How it measures](docs/how-it-measures.md)**.

## What it does not do

- **Likeness.** Colour, skin, age or expression that leave the shape alone
  are not drift; a face can measure zero and still look older.
- **Deliberate changes.** The repair moves every measured point back,
  including any the edit moved on purpose. A feature the edit redrew - bigger
  eyes, in our tries - is not measured at all: the flow cannot follow it
  there and back, so it is left out (orange crosses in the view) and left as
  the edit made it. SameFace is for the drift nobody asked for.
- **Exact figures for very small drift.** Under about 0.1% Measure reads low,
  and one feature moved a pixel or less can be missed. A repair leaves about
  0.02-0.03% however small the drift, and a fresh measurement reads that as
  0.006-0.012%: use it to see that a repair worked, not to say how much is
  left.
- **An independent check on a real edit.** The known warps have a true answer;
  a real edit has none, and it redraws texture, which a warped copy does not.
  The landmark detectors above move more on a null pair than a repair leaves,
  so they cannot check one either.
- **Turned heads.** The fits are flat, so a head turned between the two
  images, or a new expression, reads as drift.
- **Groups at once.** One face per pair: for a group, measure each person
  with their own masks.

## Quick start

1. Install **SameFace** from ComfyUI Manager, or by hand (below), and
   restart ComfyUI.
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
moved. Its outputs give the two views for a slider, the repaired image, a
shift map, a histogram and the figures; they are described in
**[The nodes](docs/nodes.md)**. It measures the surest face on each side.
Turn `repair` off to only measure.

On the Lincoln pair Simple shows about 0.35% and 0.009%, not the figures at
the top of this page, because it measures the face and ears rather than the
whole head, and shows whichever of the two fits leaves the smaller median.
Here that is the affine fit, which takes the widening out of the figure; the
repair still corrects it.

![The before with its points, the edit with arrows, the shift map and the repaired image](docs/images/simple_outputs.jpg)

*An edit that put a fan in her hand moved the face by 0.032%; repaired,
0.013%. Left to right: `before_view`, `after_view`, `shift_map`, `repaired`.*

## The advanced nodes

Every step on its own node with all its settings, for choosing one face among
several, measuring the whole head, or drawing your own plots. All are in
**[The nodes](docs/nodes.md)**.

`example_workflows/sameface_advanced.json`, with its two pictures in
`example_workflows/input/`, is a family photo edited onto a museum backdrop:
Face Mask chooses the mother's head on each side, Measure measures it, Repair
moves it back, and a second Measure measures the repair, with a view, a shift
map and a histogram of each. Her head measured 0.211%, and 0.019% after the
repair (affine fit; a head's figures, hair included, are not comparable with
a face's).

`example_workflows/sameface_demo_lincoln.json` is the workflow in the demo
video, on the same Lincoln pictures as the Quick start, measuring the whole
head with every picture node, with a SameFace Simple beside it. Expect about 0.83% and 0.33% on the edit, and
about 0.01% after the repair. It uses Set and Get nodes from
[ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes), so install that
first.

| Node | |
|---|---|
| SameFace Measure | runs the models and gives the figures and `drift`, which the others read |
| SameFace Face Mask | one person's face or head, for Measure's mask inputs |
| SameFace Repair | moves the edit's face back, keeping its framing or lining it up with the before |
| SameFace View | the before and after with the points and a mesh, for a slider |
| SameFace Shift Map | the before with each point coloured by its drift |
| SameFace Drift Histogram | how far the points drifted, after each fit |
| SameFace Drift Zones | the face as a grid, each cell's median drift |

## Open weights

No training, no API, nothing leaves the machine. The models download on first
use, about 1.8 GB, pinned to fixed revisions:

| Model | Used for | Licence |
|---|---|---|
| [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base) (base) | finding the face, ears and head from words | Apache-2.0 |
| [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) (Hiera large) | outlining them | Apache-2.0 |
| [RAFT](https://pytorch.org/vision/stable/models/raft.html) (large, torchvision) | the optical flow | BSD-3-Clause |

## Install

From **ComfyUI Manager**: search for SameFace and install.

By hand: clone into `ComfyUI/custom_nodes/` and install the requirements into
ComfyUI's environment:

```
cd ComfyUI/custom_nodes
git clone https://github.com/fixed-42/comfyui-sameface
pip install -r comfyui-sameface/requirements.txt
```

Restart ComfyUI; the nodes are under **SameFace**.

The models download automatically on first use: SAM 2.1 (0.9 GB), Grounding
DINO (0.9 GB) and RAFT (21 MB). VRAM peak usage ~3.2GiB.

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

The pictures, in this README and in `example_workflows/input/`:

- **The originals** are in the public domain: Lincoln by Alexander Gardner,
  1863 (Library of Congress, LC-DIG-ppmsca-19301), and the family from the
  Farm Security Administration/Office of War Information collection (Library
  of Congress).
- **The edits of them, and the portrait with the fan,** were made for this
  project with image models, and are released under this repository's
  licence, Apache-2.0.
- **The Lincoln edit used in the demo and the examples** came from an online
  model, in a comparison of edit models: the worst among all the candidates.
  SameFace and its workflows use open weights only.
