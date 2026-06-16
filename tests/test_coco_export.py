from pathlib import Path

from filter_sam3_detector.coco_export import convert_jsonl_to_coco
from filter_sam3_detector.utils.bbox import to_xywh


def test_extract_bbox_xywh_from_bbox_dict() -> None:
    det = {"bbox": {"x": 10, "y": 20, "width": 30, "height": 40}}
    assert to_xywh(det) == [10.0, 20.0, 30.0, 40.0]


def test_extract_bbox_xywh_from_bbox_list() -> None:
    det = {"bbox": [11, 21, 31, 41]}
    assert to_xywh(det) == [11.0, 21.0, 31.0, 41.0]


def test_extract_bbox_xywh_from_box_xyxy() -> None:
    det = {"box": [100, 200, 160, 280]}
    assert to_xywh(det) == [100.0, 200.0, 60.0, 80.0]


def test_extract_bbox_xywh_from_rois_first_entry() -> None:
    det = {"rois": [[5, 6, 25, 36]]}
    assert to_xywh(det) == [5.0, 6.0, 20.0, 30.0]


def test_extract_bbox_xywh_fallback_to_rois_on_malformed_bbox() -> None:
    det = {"bbox": {"x": 10}, "rois": [[5, 6, 25, 36]]}  # missing y, width, height
    assert to_xywh(det) == [5.0, 6.0, 20.0, 30.0]


def test_convert_jsonl_to_coco_skips_invalid_json_lines(tmp_path: Path) -> None:
    input_path = tmp_path / "detections.jsonl"
    output_path = tmp_path / "labels_coco.json"

    # First line is malformed JSON and should be skipped.
    malformed = "{this is not valid json}\n"
    valid = '{"data":{"id":1,"meta":{"detections":[{"label":"car","box":[10,20,30,50],"score":0.9}]}}}\n'
    input_path.write_text(malformed + valid, encoding="utf-8")

    coco = convert_jsonl_to_coco(input_path, output_path, "sam3_detections")

    assert len(coco["images"]) == 1
    assert len(coco["annotations"]) == 1
    assert len(coco["categories"]) == 1
    assert output_path.exists()
