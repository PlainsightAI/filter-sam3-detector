# `FILTER_REMOVE_OVERLAP` — short example

This page matches the behaviour described in **[plan-sam-stabilization.md](plan-sam-stabilization.md)**. Use it when you run **`FILTER_TEXT_PROMPTS`** with **more than one class** (e.g. `car,truck`) and the **same region** gets **two labels**; optionally remove the weaker box at **shutdown** using IoU (default **95%**) and **highest `confidence`**.

## When it applies

- **Multi-prompt only:** set **`FILTER_TEXT_PROMPTS`** (comma-separated). Single **`FILTER_TEXT_PROMPT`** does not create cross-class duplicates for this path.
- **Default:** **`FILTER_REMOVE_OVERLAP`** is **`false`** or unset → **no** cross-class removal; JSONL and overlays keep every box.
- **Opt-in:** set **`FILTER_REMOVE_OVERLAP=true`** → at shutdown the pass writes **`detections_cleaned.jsonl`** next to `detections.jsonl`, keeping one box per **cross-class** IoU cluster (winner = higher **`confidence`**). The original **`detections.jsonl`** is unchanged.

## `.env` snippet (compose-friendly)

Use with the same video and confidence as in **QUICKSTART** Example 2:

```bash
VIDEO_PATH=./data/car.mp4
FILTER_TEXT_PROMPT=
FILTER_TEXT_PROMPTS=car,truck
FILTER_CONFIDENCE_THRESHOLD=0.3

# Default: no cross-class overlap removal
# FILTER_REMOVE_OVERLAP=false

# Opt-in: remove overlapping car/truck (etc.) at shutdown when IoU ≥ threshold
FILTER_REMOVE_OVERLAP=true
```

For **local** `scripts/filter_object_detection.py`, keep **`FILTER_OUTPUT_PATH=./output/detections.jsonl`** (or your host paths) as in QUICKSTART.

## Run (Docker Compose)

```bash
docker compose -f docker-compose.yaml up -d
```

Stop when done:

```bash
docker compose -f docker-compose.yaml down
```

## What to expect

| Setting | Overlays / JSONL during run | After shutdown |
|--------|-----------------------------|----------------|
| **`FILTER_REMOVE_OVERLAP` unset or `false`** (default) | Both **`car`** and **`truck`** can appear on the **same** vehicle when the model fires both prompts | **No** extra removal step; file on disk is as written frame-by-frame |
| **`FILTER_REMOVE_OVERLAP=true`** | Same as above **until** shutdown | Shutdown pass re-reads JSONL, writes **`detections_cleaned.jsonl`**, logs **`overlap_pairs`** (before / after / removed) and **detection counts** (before / after / removed). With **`FILTER_AUTO_EXPORT_COCO=true`**, **`labels_coco.json`** is built from the **cleaned** JSONL when that file was produced this run. |

**Same-class** duplicates (e.g. two **`car`** boxes) are **not** handled by this flag — only existing **per-prompt NMS** applies.

## Verify

1. **`detections.jsonl`** exists and is non-empty.
2. With **`false`:** search a frame where the same bbox has both **`car`** and **`truck`** — both rows may be present.
3. With **`true`:** after a full stop, that frame should keep **one** row per merged cluster (the higher **`confidence`** class).
4. Full design, thresholds, and release checklist: **[plan-sam-stabilization.md](plan-sam-stabilization.md)**.

## See also

- **[QUICKSTART.md](../QUICKSTART.md)** — multi-prompt Example 2 (`car,truck` + `data/car.mp4`).
