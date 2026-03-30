"""
Cross-prompt overlap / confusion detector for multi-class SAM3 runs.

When multiple text prompts are active (e.g. ``FILTER_TEXT_PROMPTS=car,truck``),
SAM3 runs inference independently per prompt.  The same physical object can
receive **two detections with different labels** — one from each prompt — on
nearly identical bounding boxes.  This module detects those cases by computing
pairwise IoU between detections from **different** classes and flags pairs that
exceed a configurable threshold.

Same-class overlaps (e.g. two ``car`` boxes on nearby vehicles) are handled by
the existing per-prompt NMS and are **not** the concern of this module.

Usage (called from ``shutdown()`` in filter.py via ``_finalize_cross_prompt_overlaps()``):

    from .confusion_detector import ConfusionDetector

    detector = ConfusionDetector(iou_threshold=0.95)
    detections_by_class = {"car": [...], "truck": [...]}
    confusions = detector.detect(detections_by_class)
    if confusions:
        print(detector.format_warning(confusions, frame_id=42, prompts=["car", "truck"]))
"""

import logging
from itertools import combinations

__all__ = ["ConfusionDetector"]

logger = logging.getLogger(__name__)


class ConfusionDetector:
    """
    Detect cross-class detection overlaps produced by multiple text prompts.

    A *confusion event* occurs when two detections from **different** classes
    (prompts) have IoU ≥ ``iou_threshold``.  The default threshold of **0.95**
    targets near-identical boxes — i.e. the same object region labelled twice.

    Args:
        iou_threshold: Minimum IoU to flag a cross-class pair as a confusion.
                       Default 0.95 (95 %).
    """

    def __init__(self, iou_threshold: float = 0.95) -> None:
        self.iou_threshold = iou_threshold

    # ------------------------------------------------------------------
    # IoU
    # ------------------------------------------------------------------

    @staticmethod
    def compute_iou(box_a: list, box_b: list) -> float:
        """
        Compute Intersection-over-Union for two axis-aligned bounding boxes.

        Args:
            box_a: [x1, y1, x2, y2] in pixels.
            box_b: [x1, y1, x2, y2] in pixels.

        Returns:
            IoU value in [0.0, 1.0].  Returns 0.0 for degenerate boxes.
        """
        inter_x1 = max(box_a[0], box_b[0])
        inter_y1 = max(box_a[1], box_b[1])
        inter_x2 = min(box_a[2], box_b[2])
        inter_y2 = min(box_a[3], box_b[3])
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter_area

        if union <= 0.0:
            return 0.0
        return inter_area / union

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_box(det: dict) -> list | None:
        """Extract [x1, y1, x2, y2] box from a detection dict.

        Tries ``box`` (xyxy) first, then converts ``bbox`` (xywh dict or list).
        Returns None if no usable box is found.
        """
        if "box" in det:
            b = det["box"]
            if isinstance(b, (list, tuple)) and len(b) == 4:
                return [float(v) for v in b]

        bbox = det.get("bbox")
        if isinstance(bbox, dict):
            x = float(bbox.get("x", 0))
            y = float(bbox.get("y", 0))
            w = float(bbox.get("width", 0))
            h = float(bbox.get("height", 0))
            return [x, y, x + w, y + h]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = (float(v) for v in bbox)
            return [x, y, x + w, y + h]

        return None

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, detections_by_prompt: dict) -> list:
        """
        Detect cross-class confusion events.

        Args:
            detections_by_prompt: Mapping ``{class_name: [detection_dict, ...]}``.
                                  Each detection dict should have ``box`` or
                                  ``bbox``, ``score`` / ``confidence``, and
                                  optionally ``id``.

        Returns:
            List of confusion-event dicts, one per overlapping cross-class pair.
            Each event has the keys:
            ``prompt_a``, ``prompt_b``, ``iou``,
            ``box_a``, ``box_b``, ``score_a``, ``score_b``,
            ``detection_id_a``, ``detection_id_b``.
        """
        classes = list(detections_by_prompt.keys())
        if len(classes) < 2:
            return []

        confusions = []

        for class_a, class_b in combinations(classes, 2):
            dets_a = detections_by_prompt[class_a]
            dets_b = detections_by_prompt[class_b]

            for det_a in dets_a:
                box_a = self._get_box(det_a)
                if box_a is None:
                    continue
                score_a = float(det_a.get("score") or det_a.get("confidence") or 0.0)
                id_a = det_a.get("id")

                for det_b in dets_b:
                    box_b = self._get_box(det_b)
                    if box_b is None:
                        continue
                    score_b = float(det_b.get("score") or det_b.get("confidence") or 0.0)
                    id_b = det_b.get("id")

                    iou = self.compute_iou(box_a, box_b)
                    if iou >= self.iou_threshold:
                        confusions.append({
                            "prompt_a": class_a,
                            "prompt_b": class_b,
                            "iou": round(iou, 4),
                            "box_a": box_a,
                            "box_b": box_b,
                            "score_a": round(score_a, 4),
                            "score_b": round(score_b, 4),
                            "detection_id_a": id_a,
                            "detection_id_b": id_b,
                        })

        return confusions

    # ------------------------------------------------------------------
    # Warning formatting
    # ------------------------------------------------------------------

    def format_warning(self, confusions: list, frame_id, prompts: list) -> str:
        """
        Format a single actionable warning string for a set of confusion events.

        Groups events by ``(prompt_a, prompt_b)`` pair and emits tiered guidance:

        - IoU > 0.85 → merge prompts into one.
        - 0.60 < IoU ≤ 0.85 → add negative examples or adjust thresholds.
        - IoU ≤ 0.60 (just at threshold) → monitor or raise ``confusion_iou_threshold``.

        Args:
            confusions: List returned by :meth:`detect`.
            frame_id: Frame identifier for log context.
            prompts: Active prompt list (used in summary).

        Returns:
            Human-readable warning string.
        """
        if not confusions:
            return ""

        # Group by pair
        pair_events: dict = {}
        for c in confusions:
            key = (c["prompt_a"], c["prompt_b"])
            pair_events.setdefault(key, []).append(c)

        lines = [f"CONFUSION frame={frame_id}: {len(confusions)} cross-class overlap(s) detected."]

        for (pa, pb), events in pair_events.items():
            max_iou = max(e["iou"] for e in events)
            count = len(events)

            if max_iou > 0.85:
                guidance = (
                    f"'{pa}' and '{pb}' overlap at IoU={max_iou:.2f} ({count} pair(s)). "
                    "Near-identical regions — consider merging into one prompt "
                    f"(e.g. '{pa} or {pb}') or keeping only the higher-confidence class."
                )
            elif max_iou > 0.60:
                guidance = (
                    f"'{pa}' and '{pb}' overlap at IoU={max_iou:.2f} ({count} pair(s)). "
                    "Frequent overlap — add negative reference examples or raise "
                    "confidence_threshold for the weaker prompt."
                )
            else:
                guidance = (
                    f"'{pa}' and '{pb}' overlap at IoU={max_iou:.2f} ({count} pair(s)). "
                    "Mild overlap — monitor or raise confusion_iou_threshold if expected."
                )

            lines.append(f"  {guidance}")

        lines.append(
            "  To suppress: set FILTER_REMOVE_OVERLAP=true (keeps highest-confidence class per cluster) "
            "or raise FILTER_CONFUSION_IOU_THRESHOLD."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cluster-based removal (used by shutdown pass)
    # ------------------------------------------------------------------

    def remove_overlapping(self, detections: list) -> tuple:
        """
        Remove cross-class overlapping detections, keeping the highest-confidence
        detection per cluster.

        A *cluster* is formed by transitive closure: if A overlaps B and B
        overlaps C (all different classes), A, B, C form one cluster and only
        the detection with the highest ``confidence`` / ``score`` survives.

        Same-class pairs are **never** merged here.

        Tie-break (equal ``confidence``): lexicographically smaller
        ``class`` / ``label`` wins; within the same class, lower ``id`` wins.

        Args:
            detections: Flat list of detection dicts (all classes mixed).

        Returns:
            ``(kept, dropped)`` — two lists of detection dicts.
            ``kept`` + ``dropped`` cover all input detections exactly once.
        """
        if len(detections) < 2:
            return list(detections), []

        n = len(detections)

        # Build adjacency: pairs of different classes with IoU >= threshold
        edges: list[tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                det_i = detections[i]
                det_j = detections[j]
                class_i = det_i.get("class") or det_i.get("class_name") or det_i.get("label") or ""
                class_j = det_j.get("class") or det_j.get("class_name") or det_j.get("label") or ""
                if class_i == class_j:
                    continue  # same-class — handled by NMS
                box_i = self._get_box(det_i)
                box_j = self._get_box(det_j)
                if box_i is None or box_j is None:
                    continue
                if self.compute_iou(box_i, box_j) >= self.iou_threshold:
                    edges.append((i, j))

        if not edges:
            return list(detections), []

        # Union-Find for transitive clustering
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for i, j in edges:
            union(i, j)

        # Group indices by cluster root
        clusters: dict[int, list[int]] = {}
        for idx in range(n):
            root = find(idx)
            clusters.setdefault(root, []).append(idx)

        kept_indices: set[int] = set()

        for indices in clusters.values():
            if len(indices) == 1:
                # Singleton: not part of any cross-class overlap
                kept_indices.add(indices[0])
                continue

            # Check if the cluster actually has cross-class pairs
            classes_in_cluster = set()
            for idx in indices:
                d = detections[idx]
                c = d.get("class") or d.get("class_name") or d.get("label") or ""
                classes_in_cluster.add(c)

            if len(classes_in_cluster) == 1:
                # All same class — not a cross-class cluster; keep all
                kept_indices.update(indices)
                continue

            # Cross-class cluster: keep highest confidence; tie-break by class name then id
            def sort_key(idx: int):
                d = detections[idx]
                conf = float(d.get("confidence") or d.get("score") or 0.0)
                label = d.get("class") or d.get("class_name") or d.get("label") or ""
                det_id = d.get("id") or 0
                # Higher confidence first; ties: lex smaller class first; then smaller id
                return (-conf, label, det_id)

            winner = min(indices, key=sort_key)
            kept_indices.add(winner)

        kept = [detections[i] for i in range(n) if i in kept_indices]
        dropped = [detections[i] for i in range(n) if i not in kept_indices]
        return kept, dropped
