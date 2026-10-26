"""Which detections the face masker counts, at the confidence it is given.

The detector and segmenter are stood in for: the detector returns the family
photo's scores, filtered by the threshold it is asked with, and the segmenter
fills the box. That is enough to check the threshold reaches the faces and
the heads, and leaves the ears alone.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

from sameface.models import (BOX_THRESHOLD, FACE_PROMPT, FACE_THRESHOLD,  # noqa: E402
                             HEAD_PROMPT, FaceMasker, _pretrained)

POSTER = ([90.0, 50.0, 130.0, 100.0], 0.43)
BABY = ([200.0, 340.0, 290.0, 470.0], 0.59)
FOUND = {
    FACE_PROMPT: [POSTER, BABY],
    HEAD_PROMPT: [([80.0, 30.0, 140.0, 110.0], 0.34), ([185.0, 300.0, 305.0, 480.0], 0.58)],
    "ear.": [([185.0, 390.0, 205.0, 420.0], 0.26), ([285.0, 390.0, 305.0, 420.0], 0.22)],
}


class StandIn(FaceMasker):
    def __init__(self):
        super().__init__()
        self.asked = []

    def _load(self):
        return {}

    def _modules(self):
        return ()

    def detect(self, image, phrase, threshold=BOX_THRESHOLD):
        self.asked.append((phrase, threshold))
        return [found for found in FOUND[phrase] if found[1] >= threshold]

    def segment(self, rgb, box):
        mask = np.zeros(rgb.shape[:2], np.uint8)
        x0, y0, x1, y1 = (int(value) for value in box)
        mask[y0:y1, x0:x1] = 1
        return mask


RGB = np.zeros((500, 400, 3), np.uint8)


def test_measures_own_default_counts_the_poster():
    region = StandIn()(RGB, order="left to right", index=0)
    assert region.faces_found == 2
    assert region.face_score == pytest.approx(0.43)


def test_the_nodes_default_drops_the_poster_and_shifts_the_index():
    region = StandIn()(RGB, order="left to right", index=0, threshold=FACE_THRESHOLD)
    assert region.faces_found == 1
    assert region.face_score == pytest.approx(0.59)


def test_ears_keep_their_own_threshold():
    masker = StandIn()
    masker(RGB, threshold=0.5)
    assert ("ear.", BOX_THRESHOLD) in masker.asked
    assert (FACE_PROMPT, 0.5) in masker.asked


def test_a_head_is_counted_at_the_confidence_and_its_parts_by_where_they_sit():
    masker = StandIn()
    region = masker(RGB, region="head", threshold=0.5, ear_threshold=0.2)
    assert region.faces_found == 1
    assert masker.asked == [(HEAD_PROMPT, 0.5), (FACE_PROMPT, BOX_THRESHOLD),
                            ("ear.", 0.2)]


def test_the_ear_confidence_decides_which_ears_are_found():
    assert StandIn()(RGB).ears_found == 1
    assert StandIn()(RGB, ear_threshold=0.2).ears_found == 2


def test_nothing_above_the_confidence_is_an_error_that_names_it():
    with pytest.raises(ValueError, match="no face found .*at confidence 0.9"):
        StandIn()(RGB, threshold=0.9)


class HairApart(StandIn):
    """The head box's segment is the hair alone, 10px above the face."""

    def segment(self, rgb, box):
        if box == FOUND[HEAD_PROMPT][1][0]:
            mask = np.zeros(rgb.shape[:2], np.uint8)
            mask[300:330, 185:305] = 1
            return mask
        return super().segment(rgb, box)


def test_hair_across_the_gap_is_joined_to_the_face():
    region = HairApart()(RGB, region="head", gap=0.1)
    assert region.mask[310, 245] == 1 and region.mask[400, 245] == 1


def test_a_head_takes_in_ears_its_hair_does_not_cover():
    region = HairApart()(RGB, region="head", gap=0.1, ear_threshold=0.2)
    assert region.ears_found == 2 and region.ears_joined == 2
    assert region.mask[405, 188] == 1 and region.mask[405, 300] == 1


def test_a_head_without_ears_has_only_what_its_hair_covers():
    masker = HairApart()
    region = masker(RGB, region="head", gap=0.1, ears=False)
    assert region.ears_found == 0 and region.mask[405, 188] == 0
    assert "ear." not in [phrase for phrase, _ in masker.asked]


def test_hair_beyond_the_gap_is_an_error_not_a_face_alone():
    with pytest.raises(ValueError, match="widen gap"):
        HairApart()(RGB, region="head", gap=0.0)


class Pretrained:
    """Stands in for a model class: in the cache or not."""

    def __init__(self, cached):
        self.cached, self.calls = cached, []

    def from_pretrained(self, name, revision, local_files_only=False):
        self.calls.append(local_files_only)
        if local_files_only and not self.cached:
            raise OSError("not in the cache")
        return "model"


def test_a_cached_model_loads_without_asking_the_hub():
    kind = Pretrained(cached=True)
    assert _pretrained(kind, "name", "revision") == "model"
    assert kind.calls == [True]


def test_a_model_not_yet_cached_is_downloaded():
    kind = Pretrained(cached=False)
    assert _pretrained(kind, "name", "revision") == "model"
    assert kind.calls == [True, False]
