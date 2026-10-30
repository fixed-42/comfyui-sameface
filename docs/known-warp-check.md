# Checking against known warps

A null pair shows how far the measurement moves when nothing moved. That is
repeatability. It cannot show whether a real move is read at its true size, or
whether a repair really puts a feature back, because a real edit has no true
answer to compare with.

So here the answer is chosen. The Lincoln original is moved by warps of known
size and put through the real models, and the result is scored against the
warp. To run it: [Run it yourself](#run-it-yourself).

## What was done

- **The picture.** Lincoln's 1863 portrait, 1332 × 1575. Its face size is
  346.6 px, so 1% of face size is about 3.5 px.
- **Broad drift.** The whole head made wider by 3.7%, 1%, 0.3% and 0.1%, fading
  to nothing around it. This is the kind of change an edit makes.
- **One feature.** One eye moved up by 1%, 0.3% and 0.1% of face size, the rest
  of the face left alone.
- **Twice each.** As it is, and with colour, tone and sharpness changed and a
  JPEG on top, so the pixels differ as they do in an edit.
- **The real path.** The real Grounded SAM 2 and RAFT, through the same
  `measure` and `repair` the nodes run.

What a repair left is worked out two ways, neither of which runs the flow:

- **Left:** the repair's own warp is applied to where the known warp really put
  each point, at 2,850 points over the whole face, including any the flow
  dropped. This is arithmetic.
- **Lucas-Kanade:** classical tracking on the pixels, from the unwarped picture
  to the repaired one. It shares no code or model with RAFT.

## Results

All in % of face size, similarity fit, medians over the face. Your run may
differ in the last digit.

| Known change | True drift | Measure read | Share of the move read | Left | Lucas-Kanade | Fresh measurement |
|---|---|---|---|---|---|---|
| Head 3.7% wider | 0.883 | 0.875 | 1.00 | 0.033 | 0.034 | 0.007 |
| Head 1% wider | 0.242 | 0.223 | 0.99 | 0.020 | 0.023 | 0.006 |
| Head 0.3% wider | 0.073 | 0.052 | 0.88 | 0.019 | 0.018 | 0.010 |
| Head 0.1% wider | 0.024 | 0.010 | 0.67 | 0.014 | 0.020 | 0.008 |
| Eye up 1% | 0.042 | 0.029 | 1.00 | 0.015 | 0.015 | 0.003 |
| Eye up 0.3% | 0.013 | 0.005 | 0.75 | 0.008 | 0.006 | 0.001 |
| Eye up 0.1% | 0.004 | 0.001 | 0.07 | 0.004 | 0.005 | 0.001 |

With colour, tone and sharpness changed as well:

| Known change | True drift | Measure read | Share of the move read | Left | Lucas-Kanade | Fresh measurement |
|---|---|---|---|---|---|---|
| Head 3.7% wider | 0.883 | 0.868 | 1.00 | 0.033 | 0.034 | 0.006 |
| Head 1% wider | 0.242 | 0.230 | 1.01 | 0.023 | 0.027 | 0.009 |
| Head 0.3% wider | 0.073 | 0.054 | 0.88 | 0.018 | 0.016 | 0.012 |
| Head 0.1% wider | 0.024 | 0.010 | 0.60 | 0.015 | 0.023 | 0.009 |
| Eye up 1% | 0.042 | 0.028 | 0.99 | 0.015 | 0.016 | 0.003 |
| Eye up 0.3% | 0.013 | 0.005 | 0.76 | 0.009 | 0.010 | 0.002 |
| Eye up 0.1% | 0.004 | 0.002 | 0.12 | 0.005 | 0.007 | 0.002 |

**Share of the move read** is taken at the points the warp moved most, where
there is most to read: 1.00 is the whole move. **Measure read** is the
whole-face median, which is why a move of one eye looks small there: most of the
face did not move. With no warp at all, Measure read 0.001%, or 0.002% with
the colour change.

## What it shows

- **Large drift is read accurately.** At the size of the drift in the demo,
  Measure read 99% of it, and the repair removed 96%.
- **Small drift reads low.** Measure read 92% at 0.24%, 71% at 0.07% and 42% at
  0.024%. Treat figures under about 0.1% as lower bounds. The noise floor of
  0.01% says how far the measurement repeats, not how close it comes to the
  truth.
- **A repair has a limit of its own.** It left 0.014-0.033%, however small the
  drift it started from, so a repair gains little on drift under about 0.05%.
- **A fresh measurement reads a repair low.** It read 0.006-0.012% where the
  truth was 0.014-0.033%, up to five times too low. What is left is about a
  tenth of a pixel, too small for RAFT to read. Left and Lucas-Kanade agree with
  each other, and neither runs RAFT. So use a fresh measurement to see that a
  repair worked, not to say how much is left.
- **The demo.** Its repair measures about 0.010% on a real edit of this size.
  In the matching known warp a fresh measurement read 0.007% where the truth was
  0.033%, so the demo's repair probably left about 0.03%. That is an inference
  from a warped copy, not a measurement of the real edit.
- **A small move of one feature can be missed.** One eye moved 3.5 px was read
  in full, 1 px at 75%, and 0.35 px at 7%.
- **The picture agrees.** Inside the face, the mean pixel difference from the
  same picture unwarped fell from 7.0 to 0.6 out of 255 at 3.7%, and from 2.5 to
  0.5 at 1%. At 0.1% it hardly moved, 0.25 to 0.19.
- **The colour change made little difference**, which is the case a null pair
  cannot test.

## Against the marks

The marks were written into the tool before the first run. After that run,
what `Left` counts was corrected to match Measure's own definition of drift,
with the move, scale and turn taken out, and the smaller widenings and
Lucas-Kanade were added. The marks did not change.

| Mark | Passed | Failed |
|---|---|---|
| The move read within 20%, where it moved most | head 3.7%, 1%, 0.3%; eye 1% | head 0.1%; eye 0.3%, 0.1% |
| At least 90% of points kept | every case, at 100% | none |
| Under 25% of the drift left by the repair | head 3.7% (4%), 1% (8%) | head 0.3% (26%), 0.1% (56%); every eye case |

## What it cannot show

- **A real edit.** A warped copy keeps its texture, which is the easy case for
  optical flow: a real edit also redraws it. That is why every point was kept
  here, and the colour change closes only part of that gap. On a real edit the
  figures may be worse.
- **More than one portrait.** This is one picture and one warp of each kind: a
  check, not a statistic.
- **A check of a real repair.** Left needs a known warp, and Lucas-Kanade needs
  the two images to look alike, so neither applies to a real edit, which has no
  true answer and redraws what it would track.

## Run it yourself

With ComfyUI's Python, from the pack's folder:

    python tools/known_warp_check.py

It takes a few seconds a case on a GPU. The warps and the scoring are in
`sameface/known_warp.py`, and tested in `tests/test_known_warp.py`.
