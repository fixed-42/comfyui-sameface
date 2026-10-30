# How it measures

The method behind every figure, how to read a report, the study it comes from,
and what the figures are not. The nodes are in [The nodes](nodes.md).

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
   same edit measured lower once it had been moved onto the before. On 66
   edited pairs, no figure rose with the second pass, and they fell by
   14-77%.
3. **Flat fit**: a similarity (move, scale, turn) and an affine (also stretch
   and slant) are fitted through the points. What each fit leaves at a point is
   the drift there, as a percentage of face size. **Face size** is twice the
   RMS distance of the measured points from their centre, on the before.
   - Large under similarity but small under affine: the edit mostly stretched the face.
   - Large under both: features moved against each other.
4. **Repair**: a smooth warp, a thin-plate spline through the points, eased
   to nothing just beyond the face, so hair and background are left as they
   were. In the after's framing, the before's points are first carried into
   the after by the after's own pose; in the before's, the after is first
   moved by the similarity fit. The warp puts its own points where they belong
   by construction, so what it left is found only by measuring the repaired
   image again.

The noise floor is **0.01% of face size**: below it, a shift cannot be told
from the measurement's own noise. It comes from null pairs, each face resampled
to 512 and 1414 pixels and saved as a JPEG, so the true drift is zero: twice
their median drift (0.0033% over 25 faces), rounded up to a 1-2-5 step. It is
a floor for a face's median; single points scatter more. On the shift map,
grey is within that floor. To measure it on your own faces, with ComfyUI's
Python, from this folder:

    python tools/calibrate_noise_floor.py <folder of face images>


## Reading a report

The pair in the demo video: Lincoln's 1863 portrait against an edit that
colourised and restored it. Both sides are masked with SameFace Face Mask
`head`, hair included, since the edit widened the whole head. Measure's
`report` on the edit:

    SameFace drift
    RAFT, two passes, 362 points on the face; noise floor 0.01% of face size
      similarity median 0.830%  p90 1.349%  highest 1.99%  above noise 100%
                 scale 1.0908  turn +0.51 deg
      affine     median 0.330%  p90 0.536%  highest 0.97%  above noise 100%
                 scale 1.0843  turn +0.58 deg  stretch x 1.0647 y 1.1044  shear -0.11 deg

And on its repair (Repair, `framing` `after`), measured against the original
with the same head masks:

      similarity median 0.009%  p90 0.020%  highest 0.08%  above noise 44%
                 scale 1.0911  turn +0.53 deg
      affine     median 0.007%  p90 0.017%  highest 0.08%  above noise 33%
                 scale 1.0911  turn +0.53 deg  stretch x 1.0909 y 1.0912  shear +0.00 deg

- **Points**: how many the flow followed there and back, inside the masks.
  When some did not come back, a line says how many more were dropped.
- **Median**: the headline figure; half the points drifted less. The edit's
  0.830% and 0.330% are 83 and 33 times the noise floor; the repair's are
  below it.
- **p90, highest**: the 90th percentile and the single largest point.
- **Above noise**: the share of points beyond the floor - all of them on the
  edit; 44% and 33% after the repair, since the floor is for the median and
  single points scatter more.
- **Scale, turn**: how much the edit's face is scaled and turned to lie on the
  original's - here 9% larger and half a degree. Size and tilt are taken out:
  they are not drift.
- **Stretch and shear** (affine only): the scale along the edit's own x and y
  axes, and how far they lean from square. Equal stretches mean the
  proportions held. Here the edit's x is scaled by 1.0647 and its y by 1.1044
  to land on the original: the edit's head is 1.1044 / 1.0647 = 1.037 times as
  wide for its height - 3.7% wider. After the repair, 1.0909 and 1.0912: the
  proportions are back.
- **Similarity against affine**: the similarity median, 0.830%, includes that
  stretch; the affine takes it out and leaves 0.330%, still 33 times the floor.
  So besides being widened, parts of the face moved against each other - on
  the view, the arrows around the eyes show it.

Measured again in another run, the same pair gave 0.830% and 0.328%, and its
repair 0.010% and 0.008% - the Lincoln figures in the README. The figures
repeat to a few thousandths of a percent.


## The study behind it

SameFace is the measuring part of a study of one question: how far does an
edit move a face? Answering it meant first knowing how far each way of
measuring moves on its own.

**A difference image does not answer it.** An edit rewrites texture
wholesale, so every pixel changes; the difference shows skin grain, hair
detail and sharpening, not where the features went. The measurement has to
be made on points that mean the same spot in both images.

**A few points cannot measure.** A similarity fit has four degrees of freedom
and an affine six, so two points fit a similarity exactly and three fit an
affine exactly, for any two faces. A tool that aligns two faces on three
points can line them up and cannot measure them. The drift needs hundreds of
points.

**Face landmarks are too coarse.** The null test: 25 faces, each resized from
1024 px to 512 and to 1414 px and paired, so the scale between them is known
exactly and the true drift is zero. Whatever a method reports instead is its
own noise:

| | scale error | turn | median drift | 90th percentile |
|---|---|---|---|---|
| MediaPipe Face Mesh (478 points) | 0.048% | +0.022° | 0.188% | 0.565% |
| FAN 2D (68 points) | 0.036% | −0.004° | 0.644% | 1.162% |
| FAN 3D (68 points) | 0.097% | −0.073° | 0.560% | 1.140% |
| RAFT, one pass | 0.004% | −0.004° | 0.025% | 0.060% |

Drift in % of face size, as everywhere here. The RAFT row is the study's,
sampled in a face box; SameFace today, with a face mask and two passes,
measures 0.0033% on the same kind of null pairs, which is where its 0.01%
floor comes from. Most edits SameFace has measured drift 0.03-0.3%; the
demo's 0.83% is well above that. On real
before/after pairs the order was the same: RAFT about 0.15%, MediaPipe
0.57%, FAN 1.2% (one pass, before the face mask).

Why the landmarks are noisy is no mystery. FAN reads each point off a 64×64
heatmap over a crop about twice the face, so its points sit on a lattice: on
one real pair the lattice step was 2.4% of face size, against a residual of
2.0%. MediaPipe fits a mesh with a shape prior, which smooths noise - and can
smooth real shift with it - and it needs both eyes in view: on a set of hard
poses FAN found 25 faces of 25 and MediaPipe 20. And a landmark model can be
confidently wrong: on one face seen from above it fitted the mouth into the
hairline and reported nothing amiss.

**A second test, against people.** 284 faces from four public landmark sets
(FAIRSET, WFLW, 300-W and IBUG; 20,434 human-marked points), six eye and
mouth corners compared. MediaPipe's corners differed from the human marks by
a median 2.3% of the distance between the outer eye corners, FAN's by 3.9-4.0%,
and MediaPipe's outer eye corners sat inside the human marks on three faces in
four. Under 16 controlled changes - brightness, contrast, greyscale, blur,
JPEG, resampling, and known moves undone before comparing - the points moved:
the 95th percentile, per change, was 0.5-3.0% of that distance for MediaPipe
and 2.7-3.5% for FAN. Disagreeing with people is not proof of error, since
annotators have conventions too; moving when nothing moved is. No images from
those sets are in this repo.

**The comparison flatters flow a little.** Flow is measured where the image is
distinctive, while a landmark must be placed at its named spot whether or not
the pixels there decide it. The figures say flow is the better instrument for
this question, not the better face model. RAFT was trained on scenes, not
faces, which is why it is held to the null test rather than trusted.


## What the figures are not

- **Geometry, not likeness.** The drift is where the parts of the face went:
  shape and position. Changes of colour, skin texture or expression that leave
  the shape alone are not drift, and a face can measure zero drift and still
  look older or younger.
- **Not graded by an independent tool.** A repair is checked by measuring the
  repaired image again: a fresh RAFT run on new pixels, not the flow that
  built the warp, but a bias RAFT has in both runs would not show. Nothing
  independent is fine enough to check it at this scale - the landmark
  detectors above move more on a null pair than a repair leaves.
- **A floor for the median, not for each point.** Single points scatter more
  than a face's median, so on the shift map read the pattern, not one green
  point; the report's "above noise" share counts points above the same floor.
- **The null pairs change size and compression only.** An edit that also
  changes colour or light may bias the flow in ways the floor does not cover.
- **Flat fits.** A head turned between the two images, or a new expression,
  reads as drift.
- **One face per pair.** Measure measures one face on each side: the surest,
  or the one masked. For a group, measure each person with their own masks.
- **Texture the edit rewrote is left out, not measured.** Points the flow
  cannot follow there and back are dropped, and the report counts them; where
  most of a face was rewritten, few points remain. A feature redrawn on
  purpose - bigger eyes, in our tries - is among them, so the repair leaves it
  as the edit made it.
