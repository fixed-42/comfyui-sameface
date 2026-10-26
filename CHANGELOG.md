# Changelog

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
