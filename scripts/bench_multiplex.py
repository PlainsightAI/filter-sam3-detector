"""FILTER-374 benchmark: per-frame grounding latency, baseline per-prompt loop vs
multiplexed single-pass, at N in {1,2,4} prompts on the local GPU.

Baseline = the legacy path (one forward_grounding per prompt, cached text embeddings).
Multiplex = _forward_grounding_multi (one forward_grounding for all N prompts).

Backbone (set_image) is done ONCE and excluded from the timed region so the numbers
isolate the grounding-decoder cost this ticket targets.
"""

import time
from pathlib import Path

import torch
from PIL import Image

from filter_sam3_detector.filter import FilterSAM3Detector
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage

DEVICE = "cuda"
REPO = Path(__file__).resolve().parent.parent
IMG = REPO / "data" / "electrical_post.png"
# Sweetgreen prompt set (sam3-sweetgreen.yaml), truncated/padded per N.
PROMPTS = ["bowl", "salad", "avocado", "chicken", "steak", "egg", "tofu", "portobello"]
NS = [1, 2, 4]
WARMUP = 3
ITERS = 20


def build():
    bpe = REPO / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe) if bpe.exists() else None,
        device=DEVICE, eval_mode=True, load_from_HF=True,
    )
    processor = Sam3Processor(model, device=DEVICE, confidence_threshold=0.3)
    return model, processor


def make_detector(model, processor, prompts):
    d = FilterSAM3Detector.__new__(FilterSAM3Detector)
    d.model = model
    d.processor = processor
    d.device = DEVICE
    d.confidence_threshold = 0.3
    d.cached_text_embeddings = {}
    with torch.no_grad():
        for p in prompts:
            t = model.backbone.forward_text([p], device=DEVICE)
            d.cached_text_embeddings[p] = {
                "language_features": t.get("language_features"),
                "language_mask": t.get("language_mask"),
                "language_embeds": t.get("language_embeds"),
            }
    return d


def baseline_once(d, state, prompts):
    for p in prompts:
        s = {**state, "backbone_out": {**state["backbone_out"]}}
        s = d._inject_cached_text_embedding(s, p)
        d.processor.forward_grounding(s)


def multiplex_once(d, state, prompts):
    d._forward_grounding_multi(state, prompts)


def timeit(fn, d, state, prompts):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(d, state, prompts)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn(d, state, prompts)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def main():
    image = Image.open(IMG).convert("RGB")
    model, processor = build()
    print(f"{'N':>3} {'baseline_ms':>12} {'multiplex_ms':>13} {'speedup':>8}")
    for n in NS:
        prompts = PROMPTS[:n]
        d = make_detector(model, processor, prompts)
        state = processor.set_image(image)
        base = timeit(baseline_once, d, state, prompts)
        mux = timeit(multiplex_once, d, state, prompts)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{n:>3} {base:>12.2f} {mux:>13.2f} {base/mux:>7.2f}x   peakVRAM={mem:.2f}GB")
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
