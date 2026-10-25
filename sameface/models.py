"""The three models, loaded once and moved to the GPU only while in use.

- **Grounding DINO** finds the face, and the ears, from the words "face" and
  "ear".
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

from .mask_shapes import ears_beside, join, outline, parts, pick_face
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
# Low on purpose. A missed ear is a mask silently ears-out; a spurious "ear"
# is filtered by `ears_beside` and, failing that, by not touching the face.
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2


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

    def detect(self, image, phrase):
        """`(box, score)` for every place the detector saw `phrase`."""
        processor = self._loaded["detect_inputs"]
        inputs = processor(images=image, text=phrase, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._loaded["detector"](**inputs)
        found = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=BOX_THRESHOLD,
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

    def __call__(self, rgb, order="surest", index=0, ears=True):
        """The face at `index` in `order` (`mask_shapes.ORDERS`), with its
        ears joined unless `ears` is False."""
        with self:
            pil = Image.fromarray(rgb)
            faces = self.detect(pil, FACE_PROMPT)
            box, score = pick_face(faces, order, index)
            ears = ears_beside(box, self.detect(pil, EAR_PROMPT)) if ears else []
            face_mask = self.segment(rgb, box)
            ear_masks = [self.segment(rgb, ear) for ear, _ in ears]
        mask, kept = join(face_mask, ear_masks)
        return Region(mask=mask, outline=outline(mask),
                      parts=parts(face_mask, ear_masks, mask),
                      face_score=score, faces_found=len(faces),
                      ears_found=len(ears), ears_joined=int(sum(kept)))


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
