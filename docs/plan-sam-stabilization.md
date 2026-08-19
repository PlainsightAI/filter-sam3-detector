# Plan: Stabilize & Correct SAM Predictions — Confusion Detection

## Context

When **multiple** text prompts are used — via **`FILTER_TEXT_PROMPTS`** (split on **`###`**, e.g. `car###truck`) or the equivalent multi-prompt config — SAM3 runs inference **independently per prompt**. That yields **more than one class** in the same frame and the **same physical region** can get **two (or more) detections** with **different labels** (e.g. `car` and `truck` on one pickup). A single prompt (**`FILTER_TEXT_PROMPT`** only, or a single entry in `text_prompts`) does **not** create this cross-class duplication pattern.

**Default behaviour — no removal:** **`FILTER_REMOVE_OVERLAP`** defaults to **`false`** (unset counts as off). In that case there is **no** cross-class overlap **removal**: outputs keep every box the model produced; the shutdown pass may still **count** overlaps for visibility. **Removal** runs only when the operator sets **`FILTER_REMOVE_OVERLAP=true`** **and** the run has **more than one prompt / class** (see **Scope**).

### Scope — `FILTER_REMOVE_OVERLAP` only touches **different classes** that **overlap by IoU**

**In scope (removal / counting for “overlap” here):** pairs of detections where **`class` / `label` / prompt differ** (e.g. one box is **`car`**, another is **`truck`**) **and** their axis-aligned boxes have **IoU ≥ `confusion_iou_threshold`**. Default **`confusion_iou_threshold` is `0.95` (95%)** so only **near-identical** regions (same object, two labels) are merged; weaker partial overlaps keep **both** classes. For each qualifying cluster, keep **one** detection: the **class** (box) with **highest `confidence`**; drop the other(s) in that cluster only.

**Out of scope (unchanged behaviour):**

- **Only one prompt / one class** (no **`FILTER_TEXT_PROMPTS`** with multiple tokens, and not multi-prompt `prompt_sets`): there is **no** cross-class repetition from “one region, two labels”. **Do not** run removal (no-op); confusion stats can be skipped or logged as N/A.
- **Same class, multiple boxes** (e.g. two **`car`** boxes on nearby vehicles, or duplicate **`car`** on one blob): still handled exactly as today — **per-prompt NMS** and the rest of the existing pipeline. **`FILTER_REMOVE_OVERLAP` does not add a second NMS pass for same-class overlap.**
- **Single text prompt** configs: same as above — **not** the `FILTER_TEXT_PROMPTS` multi-class case.
- **Non-overlapping** boxes of different classes (IoU below threshold): **both** stay; no removal.
- **Inference, thresholds, mask extraction, JSONL format** (except optional rewrite of `meta.detections` for the cross-class overlap cases above): **unchanged**.

So: the feature **only** resolves “**same region, two different labels**” using **IoU + highest confidence**; **everything else stays the same.**

### Without `FILTER_REMOVE_OVERLAP` — what the operator sees (**default**)

**By default `FILTER_REMOVE_OVERLAP` is `false`** → **no removal** of overlapping cross-class boxes. When **`FILTER_TEXT_PROMPTS`** lists several classes, you can still see **repeated detections on the same region** (different labels); that is expected until you opt in.

When the flag is unset or not truthy (same boolean rules as **`FILTER_NMS_ENABLED`**: `true` / `1` / `yes`), the filter emits **independent** per-prompt boxes. The shutdown pass may still **count** cross-class overlaps; it **does not** drop boxes until **`FILTER_REMOVE_OVERLAP=true`** **and** there are **multiple prompts** (see **Scope**).

**What to notice** (e.g. highway with **`car` + `truck`** prompts)

- **Centre lane:** one vehicle has both a **car** box (**~0.88** confidence) and a **truck** box (**~0.66**). With **`FILTER_REMOVE_OVERLAP=true`**, the shutdown pass should keep **one** detection — **higher `confidence`** wins (here **car**).
- **Far right:** tight clusters of similar scores suggest **same-class** duplicates; those are normally handled by existing **per-prompt NMS**; this plan focuses on **cross-prompt** overlap.

**Configuration (OpenFilter):** operators use **only** **`FILTER_REMOVE_OVERLAP`** in `.env` / environment. It maps to internal **`remove_overlap`** via `FILTER_{KEY.upper()}` in `normalize_config()` (same pattern as **`FILTER_NMS_ENABLED`** → `nms_enabled`). See **Step 2**. Operator walkthrough ( **`FILTER_TEXT_PROMPTS=car###truck`**, **`FILTER_CONFIDENCE_THRESHOLD=0.3`**, **`data/car.mp4`** ): **`docs/filter-remove-overlap.md`** (also linked from **`QUICKSTART.md`**).

This plan adds:

1. **Confusion / overlap handling finalized at shutdown** — runs **before** `_run_coco_export()` when both apply: scan the written detections JSONL (`output_path`), compute cross-prompt overlap statistics, optionally write `*_cleaned.jsonl`, log **before** vs **after**. If `FILTER_REMOVE_OVERLAP` produced a cleaned file, COCO reads **that** file; otherwise the primary JSONL. **Do not** inject this pass inside `process()` (~line 1012).
2. **Optional overlap removal** — **`FILTER_REMOVE_OVERLAP`**, **default `false`** (no removal). Relevant when **`FILTER_TEXT_PROMPTS`** (or equivalent) supplies **more than one class** so the same region can get **duplicate detections** with different labels. Implement **`remove_overlap`** in `defaults` + `env_mapping` (`bool`) like **`FILTER_NMS_ENABLED`** → `nms_enabled`. When **`FILTER_REMOVE_OVERLAP=true`** **and** `len(text_prompts) > 1` (or the matching multi-prompt mode), the shutdown pass **only** removes boxes in **cross-class** clusters with IoU ≥ threshold, keeping the **highest-`confidence`** detection per cluster. **Tie-break** (identical `confidence`): deterministic rule e.g. lexicographic `class` / `label`, or smaller original index — pick one and document it.
3. **Post-processing analysis script** — reads JSONL output after the fact, aggregates confusion statistics across frames, and generates a report with resolution guidance (complements the shutdown summary).

When **`FILTER_REMOVE_OVERLAP`** is off (**default**), there is **no** cross-class box removal — only detection as today plus optional **counts** at shutdown. When **`true`** and multiple prompts are in use, the shutdown pass **also** applies removal and reports before/after counts.

---

## Files to Create / Modify

| File | Action |
|------|--------|
| `filter_sam3_detector/confusion_detector.py` | **Create** — `ConfusionDetector` class |
| `filter_sam3_detector/filter.py` | **Modify** — config, setup, `shutdown()` (overlap finalize + optional JSONL rewrite), multi-output paths |
| `filter_sam3_detector/__init__.py` | **Modify** — export `ConfusionDetector` |
| `scripts/analyze_confusions.py` | **Create** — post-processing report script |
| `docs/filter-remove-overlap.md` | **Create** — short operator example (same spirit as **`QUICKSTART.md`**): **`FILTER_TEXT_PROMPTS=car###truck`**, **`FILTER_CONFIDENCE_THRESHOLD=0.3`**, **`data/car.mp4`**, **`FILTER_REMOVE_OVERLAP=true`** vs default; expected JSONL / shutdown log |
| **`QUICKSTART.md`** | **Modify** — link to **`docs/filter-remove-overlap.md`** from the multi-prompt example; one paragraph on **`FILTER_REMOVE_OVERLAP`** |
| **`VERSION`** | **Bump** — single line, **`v` + semver** (e.g. `v0.1.8`); must match the new heading in **`RELEASE.md`** (see `pyproject.toml` → `version = { file = "VERSION" }`) |
| **`RELEASE.md`** | **Update** — add a `## vX.Y.Z - YYYY-MM-DD` section with **Added** / **Changed** / **Fixed** entries for this feature (`FILTER_REMOVE_OVERLAP`, shutdown overlap pass, optional `analyze_confusions.py`, etc.) |

**CI:** `.github/workflows/version-check.yaml` expects **`RELEASE.md`** and **`VERSION`** to stay **consistent** on every PR that ships a release-worthy change — include both in the same change set as the implementation.

---

## Step 1 — `confusion_detector.py` (new file)

New module following the same pattern as `temporal_intervals.py`.

### Class interface

```python
class ConfusionDetector:
    def __init__(self, iou_threshold: float = 0.95): ...

    @staticmethod
    def compute_iou(box_a: list, box_b: list) -> float:
        """Pure Python AABB IoU. box format: [x1, y1, x2, y2]."""

    def detect(self, detections_by_prompt: dict[str, list[dict]]) -> list[dict]:
        """
        Iterate all unordered cross-prompt pairs.
        Returns list of ConfusionEvent dicts (see below).
        Same-prompt pairs are skipped (handled by NMS already).
        """

    def format_warning(self, confusions: list[dict], frame_id, prompts: list[str]) -> str:
        """
        Groups confusions by (prompt_a, prompt_b) and returns a single
        actionable warning string with tiered guidance.
        """
```

### IoU implementation

```python
inter_x1 = max(box_a[0], box_b[0])
inter_y1 = max(box_a[1], box_b[1])
inter_x2 = min(box_a[2], box_b[2])
inter_y2 = min(box_a[3], box_b[3])
inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
union = area_a + area_b - inter_area
return inter_area / union if union > 0 else 0.0
```

### ConfusionEvent structure (JSON-serialisable)

```json
{
  "prompt_a": "car",
  "prompt_b": "truck",
  "iou": 0.97,
  "box_a": [400, 170, 520, 360],
  "box_b": [402, 172, 518, 358],
  "score_a": 0.88,
  "score_b": 0.66,
  "detection_id_a": 101,
  "detection_id_b": 107
}
```

### Tiered warning guidance in `format_warning()`

| IoU range | Message |
|-----------|---------|
| > 0.85 | "Near-identical detection regions — merge prompts into one." |
| 0.60 – 0.85 | "Frequent overlap — add negative examples or adjust `confidence_threshold`." |
| at threshold | "Mild overlap — monitor or raise `confusion_iou_threshold` if expected." |

---

## Step 2 — Config additions in `filter.py` (`normalize_config()`, ~line 130)

Add entries to the `defaults` dict:

```python
"confusion_detection_enabled": None,   # None = auto (True when >1 prompt)
"confusion_iou_threshold": 0.95,  # 95% — cross-class merge only when boxes nearly coincide
"remove_overlap": False,  # FILTER_REMOVE_OVERLAP; default false = no cross-class overlap removal
```

In **`env_mapping`**, add `"remove_overlap": bool` so **`FILTER_REMOVE_OVERLAP`** overrides `config["remove_overlap"]` with the same boolean parsing as **`FILTER_NMS_ENABLED`** (`true` / `1` / `yes`). **Default when unset: `false`** — operators must **explicitly** set **`FILTER_REMOVE_OVERLAP=true`** to enable removal. No separate env name for this feature in docs — **only** `FILTER_REMOVE_OVERLAP`.

**Note:** Removal is only meaningful when **`FILTER_TEXT_PROMPTS`** (or `prompt_sets` with multiple prompts per output) produces **more than one class**; otherwise the overlap-finalize step should **not** drop boxes (single-class runs).

Optionally wire **`FILTER_CONFUSION_IOU_THRESHOLD`** → `confusion_iou_threshold` (`float`) using the same OpenFilter pattern if operators need to tune the gate (default remains **0.95**).

Add range validation for `confusion_iou_threshold` (0.0–1.0) alongside the existing NMS threshold validation (~line 326).

---

## Step 3 — `setup()` integration in `filter.py` (~line 548)

Add a call to a new private method `_setup_confusion_detector(config)` just before the final `logger.info("FilterSAM3Detector setup complete")`.

**`_setup_confusion_detector()` logic:**
- Auto-enable overlap **detection / stats** when `len(text_prompts) > 1` OR multi-prompt `prompt_sets` — i.e. the **`FILTER_TEXT_PROMPTS`**-style case with **multiple classes** and possible **same-region duplicates**
- Single prompt → no cross-prompt confusion; **`FILTER_REMOVE_OVERLAP`** remains a no-op for removal
- Respects explicit `confusion_detection_enabled` config override
- Instantiates `ConfusionDetector(iou_threshold=confusion_iou_threshold)`
- Stores instance as `self.confusion_detector`
- Logs `info` if enabled, `debug` if disabled

---

## Step 4 — `shutdown()` integration in `filter.py` (~`shutdown()`, near `_run_coco_export`)

**Do not** run confusion overlap analysis or removal inside `process()` (~line 1012). After the JSONL file is closed, call `_finalize_cross_prompt_overlaps()` **before** `_run_coco_export()` so COCO can use `detections_cleaned.jsonl` when overlap removal runs.

**Inputs:** same `output_path` JSONL that COCO export reads (must exist and be flushed).

**Behavior:**

1. If `confusion_detection_enabled` resolves to off, **or** there is only one prompt (no **`FILTER_TEXT_PROMPTS`** multi-class case), skip overlap stats and removal (or only log N/A). Cross-class duplication arises **only** from **multiple** prompts.
2. Stream the JSONL; for each frame, count **cross-class** overlap only: group by `class` / `class_name`, then score pairs from **different** classes with IoU ≥ `confusion_iou_threshold` (confusion events). **Do not** treat two **`car`** boxes as part of this metric unless comparing to a **different** class. Accumulate **before** totals across the run (pick one metric name and use it consistently in logs).
3. If **`config["remove_overlap"]`** is true (**`FILTER_REMOVE_OVERLAP=true`**) **and** this run uses **more than one class** from **`FILTER_TEXT_PROMPTS`** (or equivalent), apply removal **per frame** using the policy below, rewrite detections in the structured records, and write back the JSONL (or emit `detections_cleaned.jsonl` — document the chosen approach). If **`remove_overlap`** is false (**default**), **never** remove boxes here.
4. Re-count overlaps on the **post-processed** result and log a clear summary for the operator, e.g.  
   `Cross-prompt overlaps: overlap_pairs before=… after=… removed=… | detections before=… after=… removed=…` (plus cleaned path when removal runs).

### Removal policy: **IoU between different classes → keep highest `confidence`**

Applies **only** when **at least two detections in the frame have different `class` / `label`** (different prompts) **and** pairwise IoU ≥ **`confusion_iou_threshold`**. Same-class pairs are **never** merged by this step.

1. Convert each detection’s `bbox` `{x, y, width, height}` to `[x1, y1, x2, y2]` for IoU.
2. Consider **only unordered pairs (A, B)** with **different** class/prompt and **IoU ≥ threshold**. Put them in the same **cluster** if they link (transitive closure).
3. In each cluster, **keep** the detection with **maximum `confidence`** (that box’s **class** wins); **remove** the other detections in that cluster **only**.
4. Detections that never appear in such a cluster are **unchanged**.

This matches the `car` / `truck` case: one region, two labels — pick the label with the better score. **Per-prompt NMS** continues to govern duplicate boxes **within** the same class.

### Example from `output/detections.jsonl` (frame `data.id` **0**)

On line 1 of that file, the same region is returned twice:

| `class` | `confidence` | `bbox` (`x`, `y`, `width`, `height`) |
|---------|----------------|----------------------------------------|
| `car`   | **~0.830**     | 642, 436, 155, 104 |
| `truck` | **~0.844**     | 642, 436, 155, 104 |

IoU is 1.0. With **`FILTER_REMOVE_OVERLAP=true`**, the shutdown pass should **drop `car` and keep `truck`** because **0.844 > 0.830**. The same frame also contains other `car` and `truck` boxes that do not overlap those pairs; only clusters above the IoU threshold are merged.

Optional: emit **tiered `logger.warning`** summaries for the worst `(prompt_a, prompt_b)` pairs **once at shutdown** instead of per frame (reduces log noise on long videos).

**Per-frame `confusions` in JSONL:** if still desired for downstream tools, it can be added by a **second pass** in this shutdown step (rewrite JSONL with `meta.confusions` filled), not from `process()`. If the team prefers zero JSONL schema change until shutdown, omit per-frame keys and rely on the analysis script only.

---

## Step 5 — `_process_multi_output()` and multiple JSONL outputs

There is **no** `process()` injection for this feature (contrast with an earlier draft). For `prompt_sets`, run the same **shutdown** post-processing **per output JSONL path** (each topic/file), since cross-set overlap remains intentionally out of scope.

---

## Step 6 — JSONL schema addition

If per-frame `confusions` are written, prefer doing so during the **shutdown** rewrite pass (not from `process()`). Frames with confusions get a `confusions` array in `meta`. Frames without confusions are **unchanged** (key omitted, not `[]`).

```json
{
  "filter_name": "SAM3Detector",
  "topic": "main",
  "data": {
    "id": 42,
    "meta": {
      "detections": ["..."],
      "confusions": [
        {
          "prompt_a": "car",
          "prompt_b": "truck",
          "iou": 0.97,
          "score_a": 0.88,
          "score_b": 0.66
        }
      ]
    }
  }
}
```

---

## Step 7 — `scripts/analyze_confusions.py` (new script)

Standalone post-processing script. No filter runtime dependency.

### CLI

```
python scripts/analyze_confusions.py detections.jsonl \
  [--iou-threshold 0.95] \
  [--output report.json] \
  [--format text|json] \
  [--min-frames 5]
```

### Processing logic

1. Stream JSONL line by line
2. If `meta.confusions` is present → use it directly
3. If not → recompute from `meta.detections` (works on historical JSONL)
4. Aggregate per `(prompt_a, prompt_b)` pair:
   - `frames_with_confusion`, `confusion_rate`, `avg_iou`, `max_iou`, `example_frame_ids`
5. Generate resolution suggestions per pair (tiered by rate + avg IoU)

### Resolution tiers

| Condition | Suggestion |
|-----------|-----------|
| `avg_iou > 0.85` and `rate > 50%` | Merge prompts into one |
| `avg_iou > 0.70` and `rate > 30%` | Add negative reference examples |
| `rate > 10%` | Raise `confidence_threshold` for the lower-scoring prompt |
| else | Acceptable overlap — consider raising `confusion_iou_threshold` |

### Text report example

```
=== Confusion Analysis Report ===
File   : output/detections.jsonl
Frames : 1847 analyzed  |  312 with confusion (16.9%)
IoU threshold: 0.95

car  vs  truck
  Confusion rate : 16.1%  (298 / 1847 frames)
  Avg IoU        : 0.88   |  Max IoU: 0.97
  Example frames : 3, 7, 12
  Suggestion     : Near-identical detection regions (avg_iou=0.88).
                   Merge into one prompt, or drop 'truck' when it loses on score (avg_score=0.66 < 'car' avg_score=0.88).

=== Recommendations ===
  1. 1 prompt pair exceeds 10% confusion rate — review before production use.
  2. Re-run with --iou-threshold 0.7 (or lower) to include partial cross-class overlaps in the report.
  3. Default filter gate is 0.95; lower `confusion_iou_threshold` only if you intentionally want to merge looser cross-class boxes.
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Auto-enable only with >1 prompt | Single-prompt configs see zero overhead and zero log noise |
| Pure Python IoU (no torch) | Detections are already Python lists post-extraction; 2–10 pairs per frame needs no vectorisation |
| Omit `confusions` key when empty | Preserves exact backward compatibility for all existing downstream consumers |
| **`FILTER_REMOVE_OVERLAP` default `false`** | **No** cross-class box removal unless the operator sets **`true`**; preserves today’s outputs by default; when off, shutdown may still emit overlap **counts** with **after = before** |
| **Removal only with multi-prompt / multi-class** | **`FILTER_TEXT_PROMPTS`** (etc.) is what creates same-region **different-label** duplicates; single prompt → removal path is a **no-op** |
| **Shutdown-only overlap pass (like COCO export)** | Avoids heavy I/O and duplicate logic in `process()`; single end-of-run summary with **before/after** overlap counts |
| **Only different-class + IoU** | Same-class overlap is **unchanged** (existing NMS); removal is strictly for **cross-class** “same region, two labels” |
| **Default IoU gate `0.95` (95%)** | Targets **almost identical** boxes; partial overlaps between classes are left as-is unless the operator lowers the threshold |
| **Winner = highest `confidence` after IoU gate** | Within a cross-class cluster above IoU threshold, keep one box — the highest `confidence` (see `output/detections.jsonl` frame 0) |
| Script re-computes if needed | Works on historical JSONL produced before this feature was added |
| Per-set scope in `prompt_sets` mode | Cross-set overlap is intentional architectural separation by output topic |

---

## Verification

### Representative use case: `car` vs `truck` on road footage

1. Run the filter on **`data/car.mp4`** with **`FILTER_TEXT_PROMPTS=car###truck`** and **`FILTER_CONFIDENCE_THRESHOLD=0.3`** (see `.env`). Expect both classes to fire on similar regions; shutdown should report non-zero **before** overlap counts when confusion detection is active.
2. With **`FILTER_REMOVE_OVERLAP`** unset or `false`, confirm shutdown logs **before** count and **after** equals **before** (no removal).
3. With **`FILTER_REMOVE_OVERLAP=true`**, confirm shutdown logs **before** and **after** with **`after` ≤ `before`** when overlaps were resolved.
4. On **`output/detections.jsonl`** (or a fresh run with `car###truck`), open frame **`data.id` 0**: for the duplicate bbox (642, 436, 155×104), after rewrite **only the `truck`** row should remain (higher **`confidence`**: ~0.844 vs ~0.830 on **`car`**).

### General checks

5. Repeat with another multi-prompt pair if needed (e.g. indoor objects); the **`car`/`truck`** highway case above is the primary regression target.
6. Verify overlap / confusion messaging appears **at shutdown** (and optionally one summary warning), not as a per-frame flood, unless explicitly designed otherwise.
7. If per-frame `confusions` are implemented via shutdown rewrite: verify `confusions` appears only on relevant frames; frames with no overlap have no `confusions` key (not `[]`).
8. Run `python scripts/analyze_confusions.py detections.jsonl` and verify report.
9. Run with a single prompt — verify no confusion-related shutdown output (or skipped as documented).
10. **Regression — same class:** on a frame with two high-IoU **`car`/`car`** (or **`truck`/`truck`**) boxes, confirm **`FILTER_REMOVE_OVERLAP=true`** does **not** remove one **solely** because of this plan’s pass (behaviour must still come from existing **per-prompt NMS** only).

### Release

11. Bump **`VERSION`** and append the matching section to **`RELEASE.md`** before merge (satisfies `version-check` workflow).

### Documentation

12. Ship **`docs/filter-remove-overlap.md`** and ensure **`QUICKSTART.md`** references it (Example 2 / `FILTER_REMOVE_OVERLAP`).
