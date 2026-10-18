"""The ComfyUI nodes: SameFace Measure, Repair and View.

Measure takes a before and an after and says how far the edit moved the face,
in figures and as a map. Repair takes what Measure found and moves the after's
features back. A repaired image is judged by measuring it again - the warp
puts its own points where they belong by construction, so only a fresh
measurement on the new pixels can say whether it helped. View draws the
measured points on the before and the after, in one frame, for a slider.

Batches: one before with any number of afters, or as many befores as afters,
paired in order.
"""
from functools import lru_cache

import numpy as np
import torch
from typing_extensions import override

import comfy.model_management
from comfy_api.latest import ComfyExtension, io

from .sameface import drawing
from .sameface.geometry import FITS
from .sameface.measure import NOISE_FLOOR, measure, repair, repair_frame
from .sameface.models import FaceMasker, Flow
from .sameface.views import EXAGGERATIONS, SHOWN, views

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
                io.Combo.Input(
                    "map_fit", options=list(FITS), default="similarity",
                    tooltip="Which fit's leftover the map shows. similarity takes out "
                            "move, scale and turn; affine also takes out stretch and "
                            "slant, so what it leaves is features moving against each other."),
            ],
            outputs=[
                Drift.Output("drift", tooltip="Everything measured, for SameFace Repair."),
                io.Image.Output("shift_map", tooltip="The before image with the drift drawn on it."),
                io.String.Output("report", tooltip="The figures in words."),
                io.Float.Output("similarity_median", tooltip="Median drift after a similarity fit, % of face size (first pair)."),
                io.Float.Output("affine_median", tooltip="Median drift after an affine fit, % of face size (first pair)."),
                io.Mask.Output("before_mask", tooltip="The face mask on the before image."),
                io.Mask.Output("after_mask", tooltip="The face mask on the after image."),
            ],
        )

    @classmethod
    def execute(cls, before, after, ears, map_fit) -> io.NodeOutput:
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
            _image([drawing.shift_map(drift, map_fit, NOISE_FLOOR) for drift in drifts]),
            text,
            drifts[0].fits["similarity"].median,
            drifts[0].fits["affine"].median,
            _mask([drift.before_region.mask for drift in drifts]),
            _mask([drift.after_region.mask for drift in drifts]),
        )


class SameFaceRepair(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SameFaceRepair",
            display_name="SameFace Repair",
            category="SameFace",
            description=(
                "Moves the after image's features back to where the before has them: "
                "the similarity fit, then a smooth warp (a thin-plate spline through the "
                "measured points, eased to nothing beyond the face). Drawn in the before "
                "image's frame, enlarged so the after's detail is kept. Measure the result "
                "again to see what it left."),
            search_aliases=["identity drift", "face restore", "fix face shift"],
            inputs=[
                Drift.Input("drift", tooltip="From SameFace Measure."),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How far along the warp goes: 0 is the flat fit alone, "
                            "1 moves every measured point all the way back."),
            ],
            outputs=[
                io.Image.Output("repaired"),
                io.Mask.Output("covered", tooltip="Where the after image reaches; outside it the repair is black."),
            ],
        )

    @classmethod
    def execute(cls, drift, strength) -> io.NodeOutput:
        frame = _shared_frame(drift)
        images, covered = [], []
        for each in drift:
            comfy.model_management.throw_exception_if_processing_interrupted()
            image, mask = repair(each, strength=strength, frame=frame)
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
                "green on the before, red on the after with an arrow to where the before has "
                "each point, and a mesh joining them - the same triangles on both, so drift "
                "shows as triangles changing shape. Feed both into Compare Images for a slider, "
                "or into Save Image to keep them. Reads what Measure found; runs no models."),
            search_aliases=["landmarks", "points overlay", "mesh overlay", "compare faces"],
            inputs=[
                Drift.Input("drift", tooltip="From SameFace Measure."),
                io.Combo.Input(
                    "after_shown", options=list(SHOWN), default="flat fit",
                    tooltip="flat fit: the after moved by the fit alone, so the arrows are the "
                            "drift. repaired: the after as Repair makes it, with its points "
                            "where the warp moved them."),
                io.Combo.Input("fit", options=list(FITS), default="similarity",
                               tooltip="The flat fit, when the flat fit is shown."),
                io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="The repair's strength, when the repair is shown."),
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
    def execute(cls, drift, after_shown, fit, strength, points, mesh,
                exaggeration, dropped) -> io.NodeOutput:
        frame = _shared_frame(drift)
        befores, afters = [], []
        for each in drift:
            first, second = views(each, frame, shown=after_shown, fit=fit, strength=strength,
                                  points=points, mesh=mesh, exaggeration=int(exaggeration),
                                  dropped=dropped, floor=NOISE_FLOOR)
            befores.append(first)
            afters.append(second)
        return io.NodeOutput(_image(befores), _image(afters))


class SameFaceExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SameFaceMeasure, SameFaceRepair, SameFaceView]


async def comfy_entrypoint() -> SameFaceExtension:
    return SameFaceExtension()
