"""Conceptual schema declarations for filter-sam3-detector (FILTER-443).

This module exists solely to exercise the FILTER-453 schema-emission +
attach path end-to-end against this filter. It is *not* wired into the
filter's runtime config (``filter.py`` still uses the legacy dict-like
``FilterConfig``) — the real FILTER-443 migration will port the runtime
config to ``FilterConfigBase`` and delete this stub.

The stub mirrors a representative slice of ``FilterSAM3DetectorConfig``'s
operator-facing surface — model + prompting + detection + NMS knobs — so
the emitted JSON Schema has enough type variety (Literal, Optional, list,
nested model, etc.) to surface real bugs in the gh-actions plumbing
rather than passing trivially.

Emit via:

    python -m openfilter.cli emit-schema --kind config filter_sam3_detector.schema
    python -m openfilter.cli emit-schema --kind output filter_sam3_detector.schema
"""

from __future__ import annotations

from typing import Literal

from openfilter.filter_runtime.config import FilterConfigBase
from openfilter.filter_runtime.output import FilterOutputSchema


class FilterSAM3DetectorSchema(FilterConfigBase):
    """Operator-facing config surface for filter-sam3-detector (stub)."""

    model_id: str = "facebook/sam3"
    device: Literal["cuda", "cpu"] = "cuda"

    text_prompt: str | None = None
    text_prompts: str | None = None
    prompt_delimiter: str = "###"
    class_delimiter: str = "|||"
    exemplars_path: str | None = None

    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5
    max_detections: int = 100

    nms_enabled: bool = True
    nms_threshold: float = 0.5

    output_masks: bool = False
    output_boxes: bool = True
    output_scores: bool = True
    output_label: str = "sam3_detections"
    output_path: str | None = None

    mixed_precision: bool = True


class SAM3Detection(FilterOutputSchema):
    """A single SAM3 detection record (helper; nested under SAM3DetectorOutput)."""

    __schema_id__ = "https://schemas.plainsight.ai/filters/sam3-detector/detection/v1"

    cls: str
    confidence: float
    bbox: list[float]
    rois: list[list[int]] | None = None


class SAM3DetectorOutput(FilterOutputSchema):
    """Top-level ``frame.data`` declaration (anchored via empty key)."""

    __schema_id__ = "https://schemas.plainsight.ai/filters/sam3-detector/v1"
    __frame_data_key__ = ""

    detections: list[SAM3Detection] = []
    width: int | None = None
    height: int | None = None
