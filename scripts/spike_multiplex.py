"""
Spike: confirm that SAM3's `model.forward_grounding` accepts a FindStage with
len(text_ids) > 1 / len(img_ids) > 1 to enable prompt-axis multiplexing
(N text prompts on one image -> single forward pass instead of N).

Compares baseline (per-prompt loop, current filter path) vs multiplex (one
batched forward) on the same image+prompts. Pass = matching boxes/scores
within numerical tolerance for each prompt.
"""

import sys
from pathlib import Path

import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage


DEVICE = "cuda"
PROMPTS = ["electrical post", "wire"]
IMG_PATH = Path(__file__).resolve().parent.parent / "data" / "electrical_post.png"
CONF_THRESHOLD = 0.5


def encode_image_and_texts(model, processor, image, prompts):
    """One image backbone pass + one text encoder pass over all prompts."""
    state = processor.set_image(image)
    text_outputs = model.backbone.forward_text(prompts, device=DEVICE)
    state["backbone_out"].update(text_outputs)
    return state


def baseline_per_prompt(model, processor, image, prompts):
    """Current filter path: loop over prompts, run forward_grounding each time."""
    results = []
    state = processor.set_image(image)
    for prompt in prompts:
        text_outputs = model.backbone.forward_text([prompt], device=DEVICE)
        # overwrite language features in place (single-prompt batch)
        state["backbone_out"].update(text_outputs)
        find_input = FindStage(
            img_ids=torch.tensor([0], device=DEVICE, dtype=torch.long),
            text_ids=torch.tensor([0], device=DEVICE, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geom = model._get_dummy_prompt(num_prompts=1)
        out = model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_input,
            find_target=None,
            geometric_prompt=geom,
        )
        # take the only batch slot
        results.append({
            "pred_logits": out["pred_logits"][0].detach().clone(),
            "pred_boxes": out["pred_boxes"][0].detach().clone(),
        })
    return results


def multiplex_one_pass(model, processor, image, prompts):
    """Spike path: encode all prompts, build one FindStage, single forward_grounding."""
    state = encode_image_and_texts(model, processor, image, prompts)
    n = len(prompts)
    find_input = FindStage(
        img_ids=torch.zeros(n, device=DEVICE, dtype=torch.long),
        text_ids=torch.arange(n, device=DEVICE, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    geom = model._get_dummy_prompt(num_prompts=n)
    out = model.forward_grounding(
        backbone_out=state["backbone_out"],
        find_input=find_input,
        find_target=None,
        geometric_prompt=geom,
    )
    return [
        {
            "pred_logits": out["pred_logits"][i].detach().clone(),
            "pred_boxes": out["pred_boxes"][i].detach().clone(),
        }
        for i in range(n)
    ]


def summarize(label, results):
    print(f"\n[{label}]")
    for i, r in enumerate(results):
        probs = r["pred_logits"].sigmoid().squeeze(-1)
        keep = probs > CONF_THRESHOLD
        print(
            f"  prompt[{i}] '{PROMPTS[i]}': "
            f"queries={tuple(r['pred_logits'].shape)}, "
            f"max_score={probs.max().item():.4f}, "
            f"top_k_above_{CONF_THRESHOLD}={keep.sum().item()}"
        )


def kept(r, threshold=CONF_THRESHOLD):
    probs = r["pred_logits"].sigmoid().squeeze(-1)
    mask = probs > threshold
    return probs[mask], r["pred_boxes"][mask]


def box_iou_cxcywh(a, b):
    # a: (N,4) cxcywh, b: (M,4) cxcywh -> (N,M)
    def to_xyxy(x):
        cx, cy, w, h = x.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)

    A = to_xyxy(a)
    B = to_xyxy(b)
    inter_lt = torch.maximum(A[:, None, :2], B[None, :, :2])
    inter_rb = torch.minimum(A[:, None, 2:], B[None, :, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def compare(a, b):
    print("\n[compare — raw tensor distance]")
    for i in range(len(a)):
        dl = (a[i]["pred_logits"] - b[i]["pred_logits"]).abs().max().item()
        db = (a[i]["pred_boxes"] - b[i]["pred_boxes"]).abs().max().item()
        print(f"  prompt[{i}]: max|dlogits|={dl:.2e}  max|dboxes|={db:.2e}")

    print("\n[compare — kept detections (post-threshold)]")
    ok = True
    for i in range(len(a)):
        sa, ba = kept(a[i])
        sb, bb = kept(b[i])
        if len(sa) != len(sb):
            print(f"  prompt[{i}]: count mismatch baseline={len(sa)} multiplex={len(sb)}")
            ok = False
            continue
        if len(sa) == 0:
            print(f"  prompt[{i}]: both empty (matches)")
            continue
        # match each baseline detection to its best multiplex peer by IoU
        ious = box_iou_cxcywh(ba, bb)
        best_iou, best_j = ious.max(dim=1)
        # pair scores in matched order
        score_diff = (sa - sb[best_j]).abs()
        print(
            f"  prompt[{i}]: pairs={len(sa)}  "
            f"min_iou={best_iou.min().item():.4f}  "
            f"max_score_diff={score_diff.max().item():.2e}"
        )
        # PASS if every baseline detection pairs with IoU>0.95 AND score diff<1e-2
        pair_ok = (best_iou.min().item() > 0.95) and (score_diff.max().item() < 1e-2)
        ok = ok and pair_ok
        if not pair_ok:
            print(f"    -> baseline scores: {sa.tolist()}")
            print(f"    -> multiplex scores (matched): {sb[best_j].tolist()}")
            print(f"    -> per-pair iou: {best_iou.tolist()}")
    return ok


def main():
    if not IMG_PATH.exists():
        print(f"ERROR: image not found at {IMG_PATH}", file=sys.stderr)
        sys.exit(1)

    image = Image.open(IMG_PATH).convert("RGB")
    print(f"Loaded image {IMG_PATH.name}: {image.size}")
    print(f"Prompts: {PROMPTS}")

    print("Building SAM3 model from HF...")
    bpe = Path(__file__).resolve().parent.parent / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe) if bpe.exists() else None,
        device=DEVICE,
        eval_mode=True,
        load_from_HF=True,
    )
    processor = Sam3Processor(model, device=DEVICE, confidence_threshold=CONF_THRESHOLD)

    with torch.no_grad():
        a = baseline_per_prompt(model, processor, image, PROMPTS)
        b = multiplex_one_pass(model, processor, image, PROMPTS)

    summarize("baseline (per-prompt loop)", a)
    summarize("multiplex (one forward)", b)
    ok = compare(a, b)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
