"""DINOv2 image embeddings with ONNX Runtime (no PyTorch needed at inference).

Model: Meta DINOv2 (Apache-2.0) exported to ONNX by the `onnx-community` organisation on
Hugging Face, pinned to an exact revision. The SAME code path (download, preprocessing,
ONNX session) is used for training-feature extraction and for inference, to avoid
train/serve skew.

Preprocessing = the model's preprocessor_config.json: RGB, resize shortest edge 256
(bicubic), center-crop 224, rescale 1/255, normalise ImageNet mean/std.
Embedding = concat(CLS token, mean of patch tokens), L2-normalised.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

from engine.config import ROOT

log = logging.getLogger(__name__)

MODELS = {
    "dinov2-small": {"repo": "onnx-community/dinov2-small", "revision": "8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"},
    "dinov2-base": {"repo": "onnx-community/dinov2-base", "revision": "31ef06cac16d5d301c5930d147002a058c85a5e4"},
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CACHE_DIR = Path(os.environ.get("MAANA_MODEL_CACHE", ROOT / "data" / "models"))


def model_path(name: str, variant: str = "model") -> Path:
    spec = MODELS[name]
    dest = CACHE_DIR / name / spec["revision"][:12] / f"{variant}.onnx"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/onnx/{variant}.onnx"
    log.warning("Downloading %s (%s) ...", name, url)
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)
    return dest


def preprocess(img: Image.Image, flip: bool = False) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    if flip:
        img = ImageOps.mirror(img)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)


class Embedder:
    def __init__(self, name: str = "dinov2-small", variant: str = "model", threads: int | None = None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.name, self.variant = name, variant
        self.session = ort.InferenceSession(str(model_path(name, variant)), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def embed_batch(self, images: list[Image.Image], tta: bool = True) -> np.ndarray:
        def run(flip: bool) -> np.ndarray:
            x = np.stack([preprocess(im, flip) for im in images]).astype(np.float32)
            out = self.session.run(None, {self.input_name: x})
            hidden = out[self.output_names.index("last_hidden_state")] if "last_hidden_state" in self.output_names else out[0]
            return np.concatenate([hidden[:, 0], hidden[:, 1:].mean(axis=1)], axis=1)

        feats = run(False)
        if tta:
            feats = (feats + run(True)) / 2
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)

    def embed(self, image: Image.Image) -> np.ndarray:
        return self.embed_batch([image])[0]


@lru_cache(maxsize=2)
def get_embedder(name: str = "dinov2-small", variant: str = "model") -> Embedder:
    return Embedder(name, variant)
