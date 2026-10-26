"""The three models, loaded once and moved to the GPU only while in use.

- **Grounding DINO** finds the face, and the ears, from the words "face" and
  "ear" - or the head, from "head".
- **SAM 2** turns each box into a mask; the ears are joined to the face.
  Ears in, neck out - the neck is out simply because nobody asked for it.
- **RAFT** measures where every pixel of the before face went in the after.

All three are open weights: Grounding DINO and SAM 2 are Apache-2.0, RAFT's
torchvision weights BSD-3-Clause. They download on first use, into the
Hugging Face and torch caches.
"""
import threading

import numpy as np
import torch
from PIL import Image

from .mask_shapes import (JOIN_REACH, REGIONS, distinct, ears_beside, face_in, join, outline,
                          parts, pick_face)
from .measure import Region

# Pinned to commits, not to `main`: the weights decide the outline, and a
# silently updated checkpoint changes every figure without changing a line.
DETECTOR = "IDEA-Research/grounding-dino-base"
DETECTOR_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
SEGMENTER = "facebook/sam2.1-hiera-large"
SEGMENTER_REVISION = "665f8e2ad61cf5f53d65644ff27c8ee525124610"

# One word each, with the full stop Grounding DINO expects after a phrase.
# Asked separately: together, the detector splits its confidence between them.
FACE_PROMPT = "face."
EAR_PROMPT = "ear."
# The whole head, hair included. "hair." alone is no use: on some images its
# box is the whole picture.
HEAD_PROMPT = "head."
# Low on purpose. A missed ear is a mask silently ears-out; a spurious "ear"
# is filtered by `ears_beside` and, failing that, by not touching the face.
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
# The Face Mask node's default for the faces or heads it counts, which the
# user can change. Measure finding its own face keeps BOX_THRESHOLD, so an
# image whose one face scores low still measures. On the family photo the
# people score 0.58-0.77, a face on a poster 0.34-0.54, nothing else over 0.13.
# The parts of a chosen person - the face inside a head, the ears - are found
# by where they sit, not counted, so they keep the low threshold; the node
# lets the user move the ears', whose scores sit right on it (0.22-0.28).
FACE_THRESHOLD = 0.5


def _pick(found, order, index, threshold, what="face"):
    """`pick_face`, with the confidence in the error: it decides the count."""
    try:
        return pick_face(found, order, index, what)
    except ValueError as error:
        raise ValueError(f"{error} at confidence {threshold:g} - lower it if a {what} "
                         f"is missed") from error


def _devices():
    """Where to run and where to rest, as ComfyUI would have it."""
    try:
        import comfy.model_management as management
        return management.get_torch_device(), management.unet_offload_device()
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.device("cpu")


class _OnDemand:
    """Loaded on first use, kept on the offload device, lent to the GPU."""

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = None

    def _load(self):
        raise NotImplementedError

    def _modules(self):
        raise NotImplementedError

    def __enter__(self):
        self._lock.acquire()
        try:
            if self._loaded is None:
                self._loaded = self._load()
            self.device, self._rest = _devices()
            for module in self._modules():
                module.to(self.device)
        except BaseException:
            self._lock.release()
            raise
        return self

    def __exit__(self, *_):
        for module in self._modules():
            module.to(self._rest)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._lock.release()


class FaceMasker(_OnDemand):
    """`masker(rgb)` -> `Region`, with Grounded SAM 2."""

    def _load(self):
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)
        return {
            "detect_inputs": AutoProcessor.from_pretrained(
                DETECTOR, revision=DETECTOR_REVISION),
            "detector": AutoModelForZeroShotObjectDetection.from_pretrained(
                DETECTOR, revision=DETECTOR_REVISION).eval(),
            "segment_inputs": Sam2Processor.from_pretrained(
                SEGMENTER, revision=SEGMENTER_REVISION),
            "segmenter": Sam2Model.from_pretrained(
                SEGMENTER, revision=SEGMENTER_REVISION).eval(),
        }

    def _modules(self):
        return (self._loaded["detector"], self._loaded["segmenter"])

    def detect(self, image, phrase, threshold=BOX_THRESHOLD):
        """`(box, score)` for every place the detector saw `phrase` with at
        least `threshold` confidence."""
        processor = self._loaded["detect_inputs"]
        inputs = processor(images=image, text=phrase, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._loaded["detector"](**inputs)
        found = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=threshold,
            text_threshold=TEXT_THRESHOLD, target_sizes=[image.size[::-1]])[0]
        return [(box.tolist(), float(score))
                for box, score in zip(found["boxes"].cpu(), found["scores"].cpu())]

    def segment(self, rgb, box):
        """What is in `box`, at the image's own resolution."""
        processor = self._loaded["segment_inputs"]
        inputs = processor(images=rgb, input_boxes=[[[float(value) for value in box]]],
                           return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._loaded["segmenter"](**inputs, multimask_output=False)
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        return (masks[0, 0].numpy() > 0).astype(np.uint8)

    def __call__(self, rgb, order="surest", index=0, ears=True, region="face",
                 threshold=BOX_THRESHOLD, gap=JOIN_REACH, ear_threshold=BOX_THRESHOLD):
        """The face at `index` in `order` (`mask_shapes.ORDERS`) - or, with
        `region="head"`, the head at `index` - with its ears joined unless
        `ears` is False. Faces and heads scoring below `threshold` are not
        counted, ears below `ear_threshold` not found; ears and hair up to
        `gap` (a share of the face's diagonal) away from the face are joined."""
        if region not in REGIONS:
            raise ValueError(f"unknown region {region!r}: one of {', '.join(REGIONS)}")
        if region == "head":
            return self._head(rgb, order, index, ears, threshold, gap, ear_threshold)
        with self:
            pil = Image.fromarray(rgb)
            faces = distinct(self.detect(pil, FACE_PROMPT, threshold))
            box, score = _pick(faces, order, index, threshold)
            face_mask = self.segment(rgb, box)
            ears = self._ears(pil, rgb, box, ear_threshold) if ears else []
        mask, kept = join(face_mask, ears, gap)
        return Region(mask=mask, outline=outline(mask),
                      parts=parts(face_mask, ears, mask),
                      face_score=score, faces_found=len(faces),
                      ears_found=len(ears), ears_joined=int(sum(kept)))

    def _ears(self, pil, rgb, face_box, threshold):
        """The masks of the ears beside `face_box`."""
        found = ears_beside(face_box, self.detect(pil, EAR_PROMPT, threshold))
        return [self.segment(rgb, ear) for ear, _ in found]

    def _head(self, rgb, order, index, ears, threshold, gap, ear_threshold):
        """The head at `index`: SAM 2 asked for a head box outlines the hair,
        and an ear only when it happens to, so the face inside that box and
        the ears beside it are segmented too and all joined - the hair across
        a gap of up to `gap`, or it is an error rather than a head mask that
        is quietly only a face."""
        with self:
            pil = Image.fromarray(rgb)
            heads = distinct(self.detect(pil, HEAD_PROMPT, threshold))
            box, score = _pick(heads, order, index, threshold, "head")
            # Found by where it is, inside the head already chosen.
            face = face_in(box, self.detect(pil, FACE_PROMPT))
            if face is None:
                raise ValueError(f"head {index} ({order}) has no face inside it")
            hair = self.segment(rgb, box)
            face_mask = self.segment(rgb, face[0])
            ears = self._ears(pil, rgb, face[0], ear_threshold) if ears else []
        mask, kept = join(face_mask, [hair, *ears], gap)
        if not kept[0]:
            raise ValueError(f"head {index} ({order}): the hair does not reach the face "
                             f"across a gap of {gap * 100:g}% - widen gap")
        return Region(mask=mask, outline=outline(mask), parts=parts(mask, [], mask),
                      face_score=score, faces_found=len(heads),
                      ears_found=len(ears), ears_joined=int(sum(kept[1:])))


class Flow(_OnDemand):
    """`flows(first, second)` -> forward and backward flow, with RAFT."""

    def _load(self):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        self.weights = Raft_Large_Weights.C_T_SKHT_V2
        return {"raft": raft_large(weights=self.weights, progress=False).eval()}

    def _modules(self):
        return (self._loaded["raft"],)

    def _tensor(self, rgb):
        # [0, 1] then into [-1, 1], which is what the weights' transforms do.
        # Feeding raw uint8 would be silently wrong rather than an error.
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        return tensor.sub_(0.5).div_(0.5).unsqueeze(0).to(self.device)

    def __call__(self, first, second):
        with self, torch.no_grad():
            first, second = self._tensor(first), self._tensor(second)
            raft = self._loaded["raft"]
            forward = raft(first, second)[-1][0].cpu().numpy()
            backward = raft(second, first)[-1][0].cpu().numpy()
        return forward, backward
