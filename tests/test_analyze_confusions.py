"""Tests for scripts/analyze_confusions.py (loaded by file path — not installed as package)."""

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parent.parent
    path = root / "scripts" / "analyze_confusions.py"
    spec = importlib.util.spec_from_file_location("analyze_confusions", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_suggest_winner_is_higher_avg_score_prompt() -> None:
    ac = _load_module()
    # car avg higher than truck → suggest dropping truck (loser), not car
    s = ac._suggest("car", "truck", rate=0.6, avg_iou=0.9, avg_sa=0.9, avg_sb=0.3)
    assert "drop 'truck'" in s
    assert "drop 'car'" not in s
    assert "'car' avg_score=0.900" in s or "avg_score=0.900" in s


def test_suggest_winner_pb_when_truck_scores_higher() -> None:
    ac = _load_module()
    s = ac._suggest("car", "truck", rate=0.6, avg_iou=0.9, avg_sa=0.2, avg_sb=0.95)
    assert "drop 'car'" in s
    assert "drop 'truck'" not in s


def test_suggest_equal_avg_tie_uses_pa_as_winner() -> None:
    ac = _load_module()
    s = ac._suggest("car", "truck", rate=0.6, avg_iou=0.9, avg_sa=0.5, avg_sb=0.5)
    # avg_sa >= avg_sb → winner=pa=car, loser=truck
    assert "drop 'truck'" in s


def test_detection_strength_zero_score_not_confidence() -> None:
    ac = _load_module()
    assert ac._detection_strength({"score": 0.0, "confidence": 0.9}) == pytest.approx(
        0.0
    )


def test_get_box_formats() -> None:
    """Test that _get_box can parse legacy xywh dicts/lists, box list, and new schema x1y1x2y2 dicts correctly."""
    from filter_sam3_detector.utils.bbox import to_xyxy

    # Legacy dict format (xywh)
    det_legacy_dict = {"bbox": {"x": 10, "y": 20, "width": 30, "height": 40}}
    assert to_xyxy(det_legacy_dict) == [10.0, 20.0, 40.0, 60.0]

    # Legacy list format (xywh)
    det_legacy_list = {"bbox": [10, 20, 30, 40]}
    assert to_xyxy(det_legacy_list) == [10.0, 20.0, 40.0, 60.0]

    # Box list format (xyxy)
    det_box = {"box": [10, 20, 40, 60]}
    assert to_xyxy(det_box) == [10.0, 20.0, 40.0, 60.0]

    # New schema dict format (x1y1x2y2)
    det_schema = {"bbox": {"x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0}}
    assert to_xyxy(det_schema) == [10.0, 20.0, 40.0, 60.0]


def test_confusions_from_record_extraction() -> None:
    """Test that _confusions_from_record extracts detections successfully from legacy and schema-compliant record structures."""
    ac = _load_module()

    # Legacy record structure (meta.sam3_detections list)
    legacy_record = {
        "data": {
            "id": 1,
            "meta": {
                "sam3_detections": [
                    {"box": [10, 20, 40, 60], "class": "car", "score": 0.9},
                    {"box": [12, 22, 38, 58], "class": "truck", "score": 0.8},
                ]
            },
        }
    }
    confusions_legacy = ac._confusions_from_record(legacy_record, iou_threshold=0.5)
    assert len(confusions_legacy) == 1
    assert confusions_legacy[0]["prompt_a"] == "car"
    assert confusions_legacy[0]["prompt_b"] == "truck"

    # New schema dict structure (detections.items)
    schema_dict_record = {
        "data": {
            "id": 1,
            "detections": {
                "items": [
                    {
                        "bbox": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
                        "label": "car",
                        "score": 0.9,
                    },
                    {
                        "bbox": {"x1": 12, "y1": 22, "x2": 38, "y2": 58},
                        "label": "truck",
                        "score": 0.8,
                    },
                ]
            },
            "meta": {},
        }
    }
    confusions_schema_dict = ac._confusions_from_record(
        schema_dict_record, iou_threshold=0.5
    )
    assert len(confusions_schema_dict) == 1
    assert confusions_schema_dict[0]["prompt_a"] == "car"
    assert confusions_schema_dict[0]["prompt_b"] == "truck"

    # New schema list structure (detections list)
    schema_list_record = {
        "data": {
            "id": 1,
            "detections": [
                {
                    "bbox": {"x1": 10, "y1": 20, "x2": 40, "y2": 60},
                    "label": "car",
                    "score": 0.9,
                },
                {
                    "bbox": {"x1": 12, "y1": 22, "x2": 38, "y2": 58},
                    "label": "truck",
                    "score": 0.8,
                },
            ],
            "meta": {},
        }
    }
    confusions_schema_list = ac._confusions_from_record(
        schema_list_record, iou_threshold=0.5
    )
    assert len(confusions_schema_list) == 1
    assert confusions_schema_list[0]["prompt_a"] == "car"
    assert confusions_schema_list[0]["prompt_b"] == "truck"
