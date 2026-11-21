# Changelog

## 0.2.0

- Licence: PolyForm Noncommercial 1.0.0 from this version. 0.1.0 and 0.1.1 were
  released under Apache-2.0 and stay available under it.
- README rewritten in a technical style: method, figures, noise floor, accuracy
  against known warps, example, notes.
- The Lincoln example uses the edit in the demo video, and
  `example_workflows/sameface_demo_lincoln.json` is the video's workflow.
- `tools/known_warp_check.py` and `docs/known-warp-check.md`: Measure and Repair
  checked against warps of known size. No node changes.

## 0.1.1

- The models load from the local cache once downloaded, with no request to
  the Hugging Face Hub per file on every start - so nothing in ComfyUI's log,
  and no network needed after the first run.
- The example workflows come with their pictures, in `example_workflows/input/`,
  and a note on how to run them, including that ComfyUI's Compare Images needs
  Modern Node Design (Nodes 2.0) for its slider.

## 0.1.0

First release: SameFace Simple, and the advanced nodes - Measure, Face Mask,
Repair, View, Shift Map, Drift Histogram and Drift Zones.
