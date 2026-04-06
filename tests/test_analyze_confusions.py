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
    assert ac._detection_strength({"score": 0.0, "confidence": 0.9}) == pytest.approx(0.0)
