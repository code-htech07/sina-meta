import io

import numpy as np
from PIL import Image, ImageFilter

from engine.acquisition.images import ImageRejected, ingest_image
from engine.preprocessing.quality import assess_quality, laplacian_variance, to_data_url


def test_real_sample_is_usable(sample_pod):
    q = assess_quality("x", sample_pod)
    assert q.usable and q.issues == []
    assert q.provenance.kind.value == "image_measurement"


def test_blur_is_detected(tmp_path, sample_pod):
    img = Image.open(sample_pod).convert("RGB").filter(ImageFilter.GaussianBlur(12))
    p = tmp_path / "blur.jpg"
    img.save(p)
    q = assess_quality("b", p)
    assert "blurry" in q.issues and not q.usable and q.retake_advice


def test_dark_and_small_images_are_rejected(tmp_path):
    p = tmp_path / "dark.png"
    Image.fromarray((np.random.rand(200, 300, 3) * 20).astype("uint8")).save(p)
    q = assess_quality("d", p)
    assert {"too_dark", "low_resolution"} <= set(q.issues)
    assert not q.usable


def test_laplacian_variance_flat_is_zero():
    assert laplacian_variance(np.full((50, 50), 128, dtype=np.uint8)) == 0.0


def test_ingest_stores_file_and_metadata(tmp_path, sample_leaf):
    meta = ingest_image(sample_leaf.read_bytes(), plant_part="young_leaf", images_dir=tmp_path)
    assert meta.width > 0 and len(meta.sha256) == 64
    assert (tmp_path / f"{meta.image_id}.jpg").exists()


def test_ingest_rejects_non_image(tmp_path):
    try:
        ingest_image(b"not an image", images_dir=tmp_path)
    except ImageRejected:
        return
    raise AssertionError("expected ImageRejected")


def test_data_url_is_jpeg_and_downscaled(sample_pod):
    from engine.acquisition.images import load_rgb

    url = to_data_url(load_rgb(sample_pod), max_side=256)
    assert url.startswith("data:image/jpeg;base64,")
    import base64

    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(img.size) == 256
