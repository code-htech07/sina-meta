"""Live tests against the REAL Meta Model API. Skipped unless a key is available.

Run: MAANA_LIVE_TESTS=1 MODEL_API_KEY=... pytest -m meta
"""

import os

import pytest

from engine.acquisition.images import load_rgb
from engine.config import get_meta_api_key, get_settings
from engine.meta_ai.client import MetaModelClient
from engine.meta_ai.muse_spark import MuseSparkAnalyzer
from engine.meta_ai.sam import SamSegmenter
from engine.preprocessing.quality import to_data_url

pytestmark = [pytest.mark.meta,
              pytest.mark.skipif(not (os.environ.get("MAANA_LIVE_TESTS") and get_meta_api_key()),
                                 reason="requires MAANA_LIVE_TESTS=1 and MODEL_API_KEY")]


def test_models_listed():
    ids = MetaModelClient().list_models()
    s = get_settings()
    assert s.muse_model in ids, f"{s.muse_model} not in {ids}"


def test_muse_minimal_text():
    res = MetaModelClient().create_response({"model": get_settings().muse_model, "input": "Reply with the word OK."})
    assert res.text.strip()


def test_muse_image_structured(sample_pod):
    res = MuseSparkAnalyzer().analyze([to_data_url(load_rgb(sample_pod))], {"experiment": "live_test"})
    assert res.output.image_assessment.is_cocoa in ("yes", "uncertain")


def test_sam_segmentation(sample_pod, tmp_path):
    res = SamSegmenter().segment(load_rgb(sample_pod), "live", prompts=["cocoa pod"], artifacts_dir=tmp_path)
    assert res.status in ("ok", "no_match"), res.error
