"""The ComfyUI nodes: SameFace Measure, Repair, Shift Map, View, Drift
Histogram and Drift Zones.

Measure takes a before and an after and says how far the edit moved the face,
in figures; every picture of it is drawn by a node of its own from Measure's
`drift`, and runs no models. Repair takes what Measure found and moves the after's
features back. A repaired image is judged by measuring it again - the warp
puts its own points where they belong by construction, so only a fresh
measurement on the new pixels can say whether it helped. View draws the
measured points on the before and the after, in one frame, for a slider.
Shift Map colours each point on the before by its drift; Drift Histogram and
Drift Zones draw the points as a chart and as a grid over the face.

Batches: one before with any number of afters, or as many befores as afters,
paired in order.
"""
import json
from functools import lru_cache

import numpy as np
import torch
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, io

from .sameface import drawing
from .sameface.geometry import FITS
from .sameface.measure import FRAMINGS, NOISE_FLOOR, measure, repair, repair_frame
from .sameface.models import FaceMasker, Flow
from .sameface.plots import drift_histogram, drift_zones, point_table, same_size
from .sameface.views import EXAGGERATIONS, views

Drift = io.Custom("SAMEFACE_DRIFT")


@lru_cache(maxsize=1)
def _models():
    return FaceMasker(), Flow()


def _rgb(images):
    """A ComfyUI IMAGE batch as a list of uint8 RGB arrays."""
    array = (images.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round()
    return [np.ascontiguousarray(image[..., :3].astype(np.uint8)) for image in array]


def _image(arrays):
    return torch.from_numpy(np.stack(arrays).astype(np.float32) / 255.0)


def _mask(arrays):
    return torch.from_numpy(np.stack(arrays).astype(np.float32))


def _paired(befores, afters):
    if len(befores) == 1:
        return [(befores[0], after) for after in afters]
    if len(befores) == len(afters):
        return list(zip(befores, afters))
    raise ValueError(f"{len(befores)} befores and {len(afters)} afters: give one before, "
                     "or as many befores as afters")


def _shared_frame(drifts):
    """One frame for a batch, so the results stack: the largest any of them
    asks for, among afters drawn against befores of one size."""
    if len({drift.before_size for drift in drifts}) > 1:
        raise ValueError("the befores in this batch differ in size, so their pictures "
                         "cannot share a batch; give them one at a time")
    return max((repair_frame(drift) for drift in drifts), key=lambda size: size[0] * size[1])


class SameFaceMeasure(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SameFaceMeasure",
            display_name="SameFace Measure",
            category="SameFace",
            description=(
                "How far an edit moved the face: where every part of the before face "
                "went in the after (RAFT optical flow inside a Grounded SAM 2 face mask), "
                "with the move, scale and turn of the whole face taken out. What is left "
                "is drift, as % of face size."),
            search_aliases=["identity drift", "face shift", "face consistency"],
            inputs=[
                io.Image.Input("before", tooltip="The image before the edit."),
                io.Image.Input("after", tooltip="The edited image, or a batch of them."),
                io.Boolean.Input(
                    "ears", default=True,
                    tooltip="Count the points on the ears as well as the face."),
            ],
            outputs=[
                Drift.Output("drift", tooltip="Everything measured: for Repair, and for "
                                              "the nodes that draw it."),
                io.String.Output("report", tooltip="The figures in words."),
                io.Float.Output("similarity_median", tooltip="Median drift after a similarity fit, % of face size (first pair)."),
                io.Float.Output("affine_median", tooltip="Median drift after an affine fit, % of face size (first pair)."),
                io.Mask.Output("before_mask", tooltip="The face mask on the before image."),
                io.Mask.Output("after_mask", tooltip="The face mask on the after image."),
                io.String.Output(
                    "points",
                    tooltip="Every measured point as JSON, one list per column: where it is "
                            "on the before (x, y, pixels), how far each fit leaves it (dx, dy "
                            "in pixels; drift in % of face size), confidence and part. "
                            "A pair column numbers the pairs in a batch."),
            ],
        )

    @classmethod
    def execute(cls, before, after, ears) -> io.NodeOutput:
        masker, flows = _models()
        drifts = []
        for first, second in _paired(_rgb(before), _rgb(after)):
            comfy.model_management.throw_exception_if_processing_interrupted()
            drifts.append(measure(first, second, masker, flows, ears=ears))

        text = "\n\n".join(
            drawing.report(drift, NOISE_FLOOR,
                           f"pair {number} of {len(drifts)}" if len(drifts) > 1 else None)
            for number, drift in enumerate(drifts, 1))
        return io.NodeOutput(
            drifts,
            text,
            drifts[0].fits["similarity"].median,
            drifts[0].fits["affine"].median,
            _mask([drift.before_region.mask for drift in drifts]),
            _mask([drift.after_region.mask for drift in drifts]),
            json.dumps(point_table(drifts)),
        )


class SameFaceRepair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SameFaceRepair",
            display_name="SameFace Repair",
            category="SameFace",
            description=(
                "Moves the after image's features back to where the before has them, "
                "with a smooth warp (a thin-plate spline through the measured points, "
                "eased to nothing beyond the face). By default the after keeps its own "
                "framing and size and only the face changes. Measure the result against "
                "the before, and View that, to see what it left."),
            search_aliases=["identity drift", "face restore", "fix face shift"],
            inputs=[
                Drift.Input("drift", tooltip="From SameFace Measure."),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How far along the warp goes: 0 leaves the features where "
                            "they are, 1 moves every measured point all the way back."),
                io.Combo.Input(
                    "framing", options=list(FRAMINGS), default="after",
                    tooltip="after: the edit's own framing and size, only the face "
                            "changed. before: the whole after lined up with the before "
                            "(moved, scaled and turned), enlarged so its detail is kept; "
                            "where it does not reach is black."),
            ],
            outputs=[
                io.Image.Output("repaired"),
                io.Mask.Output("covered", tooltip="Where the after image reaches. All of it "
                                                  "with after framing; with before framing, "
                                                  "outside it the repair is black."),
            ],
        )

    @classmethod
    def execute(cls, drift, strength, framing) -> io.NodeOutput:
        frame = _shared_frame(drift) if framing == "before" else None
        images, covered = [], []
        for each in drift:
            comfy.model_management.throw_exception_if_processing_interrupted()
            image, mask = repair(each, strength=strength, frame=frame, framing=framing)
            images.append(image)
            covered.append(mask)
        return io.NodeOutput(_image(images), _mask(covered))


class SameFaceView(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SameFaceView",
            display_name="SameFace View",
            category="SameFace",
            description=(
                "The before and the after in one frame, with the measured points drawn on: "
                "green on the before, red on the after (moved by the fit) with an arrow to "
                "where the before has each point, and a mesh joining them - the same "
                "triangles on both, so drift shows as triangles changing shape. Feed both "
                "into Compare Images for a slider. To see a repair, measure it against the "
                "before and view that. Reads what Measure found; runs no models."),
            search_aliases=["landmarks", "points overlay", "mesh overlay", "compare faces"],
            inputs=[
                Drift.Input("drift", tooltip="From SameFace Measure."),
                io.Combo.Input("fit", options=list(FITS), default="similarity",
                               tooltip="The fit that moves the after onto the before; the "
                                       "arrows are what it leaves."),
                io.Boolean.Input("points", default=True),
                io.Boolean.Input("mesh", default=True),
                io.Combo.Input(
                    "exaggeration", options=[str(value) for value in EXAGGERATIONS], default="1",
                    tooltip="Stretches the arrows only; the dots and the mesh stay where the "
                            "points are. Printed on the picture."),
                io.Boolean.Input(
                    "dropped", default=True,
                    tooltip="Mark, on the before, the points left out because the flow did "
                            "not come back to where it started - the gaps in the grid."),
            ],
            outputs=[
                io.Image.Output("before_view"),
                io.Image.Output("after_view"),
            ],
        )

    @classmethod
    def execute(cls, drift, fit, points, mesh, exaggeration, dropped) -> io.NodeOutput:
        frame = _shared_frame(drift)
        befores, afters = [], []
        for each in drift:
            first, second = views(each, frame, fit=fit,
                                  points=points, mesh=mesh, exaggeration=int(exaggeration),
                                  dropped=dropped, floor=NOISE_FLOOR)
            befores.append(first)
            afters.append(second)
        return io.NodeOutput(_image(befores), _image(afters))


class SameFaceShiftMap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SameFaceShiftMap",
            display_name="SameFace Shift Map",
            category="SameFace",
            description=(
                "The before image with every measured point coloured by how far the fit "
                "left it: grey within the noise floor, then greens on a fixed scale, so a "
                "colour is the same drift on every pair. Reads what Measure found; runs "
                "no models."),
            search_aliases=["drift map", "shift heatmap"],
            inputs=[
                Drift.Input("drift", tooltip="From SameFace Measure."),
                io.Combo.Input(
                    "fit", options=list(FITS), default="similarity",
                    tooltip="Which fit's leftover the map shows. similarity takes out "
                            "move, scale and turn; affine also takes out stretch and "
                            "slant, so what it leaves is features moving against each other."),
            ],
            outputs=[io.Image.Output("shift_map")],
        )

    @classmethod
    def execute(cls, drift, fit) -> io.NodeOutput:
        return io.NodeOutput(_image(same_size([drawing.shift_map(each, fit, NOISE_FLOOR)
                                               for each in drift])))


class SameFaceHistogram(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SameFaceHistogram",
            display_name="SameFace Drift Histogram",
            category="SameFace",
            description=(
                "How many of the measured points drifted how far, after the similarity "
                "fit (bars) and the affine fit (outline), with each median and the noise "
                "floor marked. Reads what Measure found; runs no models."),
            search_aliases=["drift chart", "drift distribution", "plot"],
            inputs=[Drift.Input("drift", tooltip="From SameFace Measure.")],
            outputs=[io.Image.Output("histogram")],
        )

    @classmethod
    def execute(cls, drift) -> io.NodeOutput:
        return io.NodeOutput(_image(same_size([drift_histogram(each, NOISE_FLOOR)
                                               for each in drift])))


class SameFaceZones(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SameFaceZones",
            display_name="SameFace Drift Zones",
            category="SameFace",
            description=(
                "The before face cut into a grid, each cell tinted by the median drift of "
                "the points in it and the figure printed, in the shift map's colours. The "
                "cells are a grid over the face, not named features. Reads what Measure "
                "found; runs no models."),
            search_aliases=["drift by region", "drift heatmap", "plot"],
            inputs=[
                Drift.Input("drift", tooltip="From SameFace Measure."),
                io.Combo.Input("fit", options=list(FITS), default="similarity",
                               tooltip="Whose leftover drift the cells show."),
                io.Int.Input("rows", default=4, min=1, max=12),
                io.Int.Input("cols", default=3, min=1, max=12),
            ],
            outputs=[io.Image.Output("zones")],
        )

    @classmethod
    def execute(cls, drift, fit, rows, cols) -> io.NodeOutput:
        return io.NodeOutput(_image(same_size([drift_zones(each, fit, NOISE_FLOOR, rows, cols)
                                               for each in drift])))


class SameFaceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SameFaceMeasure, SameFaceRepair, SameFaceShiftMap, SameFaceView,
                SameFaceHistogram, SameFaceZones]


async def comfy_entrypoint() -> SameFaceExtension:
    return SameFaceExtension()
