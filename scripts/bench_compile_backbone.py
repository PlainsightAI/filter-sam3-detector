"""
Bench FILTER-373: torch.compile on the SAM3 vision backbone.

Times the backbone pass (processor.set_image, ~70% of inference) at 480p and
1080p, eager vs compiled, on the local GPU. The compiled run discards warmup
iterations (first call triggers compilation). Reports per-frame ms and speedup.

Usage: uv run python scripts/bench_compile_backbone.py
"""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEVICE = "cuda"
if not torch.cuda.is_available():
    raise SystemExit("This benchmark requires a CUDA GPU")
RESOLUTIONS = {"480p": (854, 480), "1080p": (1920, 1080)}
WARMUP = 3
ITERS = 20


def make_image(w, h):
    # Deterministic gradient image so runs are reproducible
    base = np.indices((h, w)).sum(0)
    rgb = np.stack([base % 256, (base * 2) % 256, (base * 3) % 256], axis=-1).astype("uint8")
    return Image.fromarray(rgb, "RGB")


def time_backbone(processor, image):
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        processor.set_image(image)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        processor.set_image(image)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1000.0


def build(compile_backbone):
    bpe = Path(__file__).resolve().parent.parent / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe) if bpe.exists() else None,
        device=DEVICE, eval_mode=True, load_from_HF=True, compile=compile_backbone
    )
    return Sam3Processor(model, device=DEVICE, confidence_threshold=0.5)


def main():
    rows = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for label, compile_backbone in [("eager", False), ("compiled", True)]:
            proc = build(compile_backbone)
            for res, (w, h) in RESOLUTIONS.items():
                ms = time_backbone(proc, make_image(w, h))
                rows.append((label, res, ms))
                print(f"{label:9s} {res:6s}  {ms:7.2f} ms/frame")
            del proc
            torch.cuda.empty_cache()

    print("\n=== speedup (eager / compiled) ===")
    for res in RESOLUTIONS:
        e = next(m for l, r, m in rows if l == "eager" and r == res)
        c = next(m for l, r, m in rows if l == "compiled" and r == res)
        print(f"{res:6s}  eager {e:7.2f}  compiled {c:7.2f}  speedup {e / c:.2f}x")


if __name__ == "__main__":
    main()
