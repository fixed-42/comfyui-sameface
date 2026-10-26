# How it measures

The method behind every figure. The nodes are in [The nodes](nodes.md).

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


## What the figures are not

- **Geometry, not likeness.** The drift is where the parts of the face went:
  shape and position. Changes of colour, skin texture or expression that leave
  the shape alone are not drift, and a face can measure zero drift and still
  look older or younger.
- **One face per pair.** Measure measures one face on each side: the surest,
  or the one masked. For a group, measure each person with their own masks.
- **Texture the edit rewrote is left out, not measured.** Points the flow
  cannot follow there and back are dropped, and the report counts them; where
  most of a face was rewritten, few points remain.
