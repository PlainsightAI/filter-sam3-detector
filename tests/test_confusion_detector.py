"""Unit tests for ConfusionDetector (IoU, clustering, removal, score handling)."""

import pytest

from filter_sam3_detector.confusion_detector import ConfusionDetector

# Shared near-identical boxes (IoU = 1.0)
BOX_XY = [0.0, 0.0, 100.0, 100.0]


class TestComputeIou:
    def test_identical_boxes_iou_one(self) -> None:
        b = [0.0, 0.0, 10.0, 10.0]
        assert ConfusionDetector.compute_iou(b, b) == pytest.approx(1.0)

    def test_disjoint_boxes_iou_zero(self) -> None:
        a = [0.0, 0.0, 10.0, 10.0]
        b = [20.0, 20.0, 30.0, 30.0]
        assert ConfusionDetector.compute_iou(a, b) == pytest.approx(0.0)

    def test_zero_area_box_returns_zero(self) -> None:
        degenerate = [5.0, 5.0, 5.0, 20.0]
        normal = [0.0, 0.0, 10.0, 10.0]
        assert ConfusionDetector.compute_iou(degenerate, normal) == pytest.approx(0.0)

    def test_both_zero_area_union_zero_returns_zero(self) -> None:
        z = [1.0, 1.0, 1.0, 1.0]
        assert ConfusionDetector.compute_iou(z, z) == pytest.approx(0.0)


class TestDetectionStrength:
    def test_score_zero_not_replaced_by_confidence(self) -> None:
        det = {"score": 0.0, "confidence": 0.9}
        assert ConfusionDetector._detection_strength(det) == pytest.approx(0.0)

    def test_confidence_zero_when_score_missing(self) -> None:
        det = {"confidence": 0.0}
        assert ConfusionDetector._detection_strength(det) == pytest.approx(0.0)

    def test_score_preferred_when_both_present(self) -> None:
        det = {"score": 0.4, "confidence": 0.9}
        assert ConfusionDetector._detection_strength(det) == pytest.approx(0.4)


class TestDetect:
    def test_fewer_than_two_classes_returns_empty(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        assert d.detect({}) == []
        assert d.detect({"car": []}) == []
        assert d.detect({"car": [{"box": BOX_XY, "score": 0.9}]}) == []

    def test_two_classes_no_overlap_below_threshold(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        car = {"box": [0, 0, 10, 10], "score": 0.9, "class": "car"}
        truck = {"box": [100, 100, 110, 110], "score": 0.9, "class": "truck"}
        assert d.detect({"car": [car], "truck": [truck]}) == []

    def test_cross_class_high_iou_emits_confusion(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        car = {"box": list(BOX_XY), "score": 0.9, "class": "car"}
        truck = {"box": list(BOX_XY), "score": 0.8, "class": "truck"}
        conf = d.detect({"car": [car], "truck": [truck]})
        assert len(conf) == 1
        assert conf[0]["prompt_a"] == "car"
        assert conf[0]["prompt_b"] == "truck"
        assert conf[0]["iou"] == pytest.approx(1.0)

    def test_missing_box_skipped(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        car = {"score": 0.9, "class": "car"}
        truck = {"box": list(BOX_XY), "score": 0.8, "class": "truck"}
        assert d.detect({"car": [car], "truck": [truck]}) == []


class TestRemoveOverlapping:
    def test_empty_or_single_returns_all_kept(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        assert d.remove_overlapping([]) == ([], [])
        one = [{"box": BOX_XY, "score": 0.9, "label": "car"}]
        assert d.remove_overlapping(one) == (one, [])

    def test_no_cross_class_edges_all_kept(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        dets = [
            {"box": [0, 0, 10, 10], "score": 0.9, "label": "car"},
            {"box": [50, 50, 60, 60], "score": 0.8, "label": "truck"},
        ]
        kept, dropped = d.remove_overlapping(dets)
        assert len(kept) == 2 and dropped == []

    def test_same_class_overlap_not_merged(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        dets = [
            {"box": list(BOX_XY), "score": 0.9, "label": "car", "id": 1},
            {"box": list(BOX_XY), "score": 0.8, "label": "car", "id": 2},
        ]
        kept, dropped = d.remove_overlapping(dets)
        assert len(kept) == 2 and dropped == []

    def test_transitive_cluster_keeps_highest_strength(self) -> None:
        """A↔B and B↔C (same box) → one cluster; highest score wins."""
        d = ConfusionDetector(iou_threshold=0.95)
        dets = [
            {"box": list(BOX_XY), "score": 0.9, "class": "car", "id": 0},
            {"box": list(BOX_XY), "score": 0.8, "class": "truck", "id": 1},
            {"box": list(BOX_XY), "score": 0.7, "class": "bus", "id": 2},
        ]
        kept, dropped = d.remove_overlapping(dets)
        assert len(kept) == 1 and len(dropped) == 2
        assert kept[0]["class"] == "car"
        assert kept[0]["score"] == pytest.approx(0.9)

    def test_tie_equal_strength_lexicographic_class_wins(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        dets = [
            {"box": list(BOX_XY), "score": 0.5, "class": "truck", "id": 0},
            {"box": list(BOX_XY), "score": 0.5, "class": "car", "id": 99},
        ]
        kept, dropped = d.remove_overlapping(dets)
        assert len(kept) == 1
        assert kept[0]["class"] == "car"

    def test_same_class_duplicate_boxes_both_kept(self) -> None:
        """Same-class overlaps are not merged here (per-prompt NMS domain)."""
        d = ConfusionDetector(iou_threshold=0.95)
        dets = [
            {"box": list(BOX_XY), "score": 0.5, "class": "car", "id": 2},
            {"box": list(BOX_XY), "score": 0.5, "class": "car", "id": 1},
        ]
        kept, dropped = d.remove_overlapping(dets)
        assert len(kept) == 2 and dropped == []

    def test_degenerate_boxes_no_edge_all_kept(self) -> None:
        d = ConfusionDetector(iou_threshold=0.95)
        line = [10.0, 10.0, 10.0, 50.0]
        dets = [
            {"box": line, "score": 0.9, "class": "car"},
            {"box": line, "score": 0.8, "class": "truck"},
        ]
        kept, dropped = d.remove_overlapping(dets)
        assert len(kept) == 2 and dropped == []

    def test_score_zero_beats_confidence_via_strength_order(self) -> None:
        """score=0.0 must not fall through to confidence when both keys exist."""
        d = ConfusionDetector(iou_threshold=0.95)
        dets = [
            {"box": list(BOX_XY), "score": 0.0, "confidence": 0.99, "class": "car"},
            {"box": list(BOX_XY), "score": 0.1, "confidence": 0.1, "class": "truck"},
        ]
        kept, dropped = d.remove_overlapping(dets)
        assert len(kept) == 1
        assert kept[0]["class"] == "truck"


def test_format_warning_empty() -> None:
    d = ConfusionDetector()
    assert d.format_warning([], frame_id=1, prompts=["a", "b"]) == ""
