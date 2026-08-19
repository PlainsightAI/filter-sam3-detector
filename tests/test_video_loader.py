"""Tests for the vendored SAM3 video loader after the decord -> PyAV swap.

These import the *vendored* ``sam3/sam3/model/utils/sam2_utils.py`` directly by
path (via importlib), not the installed ``sam3`` wheel: ``make install`` uses pip,
which ignores ``[tool.uv.sources]`` and pulls the published wheel whose copy of
this file still ``import decord``s. Loading the file directly guarantees we
exercise the PyAV implementation this repo ships to the Docker image.

A backend swap is exactly the change where a silent RGB/BGR flip or a
height/width transposition would still produce plausible detections, so we assert
shape, dtype, native dimensions (in order), channel order, and bytes-input parity.
"""

import importlib.util
import pathlib

import numpy as np
import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDORED = REPO_ROOT / "sam3" / "sam3" / "model" / "utils" / "sam2_utils.py"
CAR_MP4 = REPO_ROOT / "data" / "car.mp4"
CPU = torch.device("cpu")

# ImageNet mean/std the loader applies by default — invert to recover ~[0,1] RGB.
_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _load_video_frames_from_video_file():
    spec = importlib.util.spec_from_file_location("vendored_sam2_utils", VENDORED)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_video_frames_from_video_file


pytestmark = pytest.mark.skipif(not CAR_MP4.exists(), reason="data/car.mp4 not present")


def test_shape_dtype_and_native_dimensions():
    cv2 = pytest.importorskip("cv2")
    cap = cv2.VideoCapture(str(CAR_MP4))
    exp_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    exp_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    exp_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    load = _load_video_frames_from_video_file()
    images, height, width = load(
        str(CAR_MP4), image_size=64, offload_video_to_cpu=True, compute_device=CPU
    )

    assert images.dtype == torch.float32
    assert images.ndim == 4 and images.shape[1:] == (3, 64, 64)  # (T, C, H, W), square-resized
    # Pin the exact decoded frame count against cv2's reference. A backend swap
    # can silently drop/duplicate frames (car.mp4 is 512 frames, ~68% B-frames —
    # exactly where reorder mishandling shows up as an off-by-a-handful count),
    # and nothing else in this file would catch that.
    assert exp_frames > 0 and images.shape[0] == exp_frames
    # Native dims reported height-then-width, not transposed.
    assert (height, width) == (exp_h, exp_w)


def test_bytes_input_matches_path_input():
    load = _load_video_frames_from_video_file()
    from_path, hp, wp = load(str(CAR_MP4), 32, offload_video_to_cpu=True, compute_device=CPU)
    from_bytes, hb, wb = load(CAR_MP4.read_bytes(), 32, offload_video_to_cpu=True, compute_device=CPU)

    assert from_path.shape == from_bytes.shape
    assert (hp, wp) == (hb, wb)
    assert torch.allclose(from_path, from_bytes)


def test_channel_order_is_rgb_not_bgr():
    cv2 = pytest.importorskip("cv2")
    load = _load_video_frames_from_video_file()
    images, _, _ = load(str(CAR_MP4), 64, offload_video_to_cpu=True, compute_device=CPU)
    first_rgb = images[0] * _STD + _MEAN  # invert normalization -> ~[0,1], channels = R,G,B

    cap = cv2.VideoCapture(str(CAR_MP4))
    ok, bgr = cap.read()
    cap.release()
    assert ok, "cv2 could not read the reference first frame"
    ref = cv2.cvtColor(cv2.resize(bgr, (64, 64)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref = torch.from_numpy(ref).permute(2, 0, 1)

    def corr(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-8))

    aligned = np.mean([corr(first_rgb[c], ref[c]) for c in range(3)])
    swapped = np.mean([corr(first_rgb[c], ref[2 - c]) for c in range(3)])
    # An RGB/BGR flip would make the swapped mapping correlate better.
    assert aligned > swapped, f"channel order looks flipped: aligned={aligned:.3f} swapped={swapped:.3f}"
