"""Adapter tests against the documented wire contract, using httpx.MockTransport.

These mocks test OUR parsing/request-building code; they do not claim that the API
works. Live checks are in tests/integration/test_meta_live.py (requires a key).
"""

import json

import httpx
import numpy as np
import pytest
from PIL import Image

from engine.meta_ai.client import MetaAuthError, MetaModelClient, MetaNotConfiguredError, extract_output_text
from engine.meta_ai.muse_spark import MuseSparkAnalyzer, parse_muse_output
from engine.meta_ai.sam import SamSegmenter, parse_sam_output

FAKE_KEY = "test-key-0123456789abcdef"


def _response(text: str) -> dict:
    return {"id": "resp_1", "object": "response", "status": "completed",
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 1000, "output_tokens": 500, "output_tokens_details": {"reasoning_tokens": 200},
                      "input_tokens_details": {"cached_tokens": 0}}}


def test_extract_output_text_ignores_reasoning_items():
    assert extract_output_text(_response("hello")) == "hello"


def test_not_configured_raises():
    with pytest.raises(MetaNotConfiguredError):
        MetaModelClient(api_key="").create_response({"model": "m", "input": "x"})


def test_auth_error_does_not_leak_key():
    def handler(request):
        assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"
        return httpx.Response(401, json={"error": {"message": f"bad key Bearer {FAKE_KEY}"}})
    client = MetaModelClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler), max_retries=0)
    with pytest.raises(MetaAuthError) as exc:
        client.create_response({"model": "m", "input": "x"})
    assert FAKE_KEY not in str(exc.value)


def test_optional_param_dropped_and_reported():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if "text" in body:
            return httpx.Response(400, json={"error": {"message": "unsupported parameter: text.format"}})
        return httpx.Response(200, json=_response("{}"))
    client = MetaModelClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler), max_retries=0)
    res = client.create_response({"model": "m", "input": "x", "text": {"format": {"type": "json_schema"}}}, optional_keys=("text",))
    assert res.dropped_params == ["text"] and len(seen) == 2


def test_muse_payload_follows_contract(muse_fixture):
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=_response(json.dumps(muse_fixture)))
    analyzer = MuseSparkAnalyzer(client=MetaModelClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler)))
    res = analyzer.analyze(["data:image/jpeg;base64,AAAA"], {"weather": {"status": "ok"}})
    assert captured["model"] == "muse-spark-1.3"
    content = captured["input"][0]["content"]
    assert captured["input"][0]["role"] == "user"
    assert {"type": "input_image", "image_url": "data:image/jpeg;base64,AAAA"} in content
    assert captured["reasoning"] == {"effort": "medium"}
    assert captured["text"]["format"]["type"] == "json_schema" and captured["text"]["format"]["strict"] is True
    schema = captured["text"]["format"]["schema"]
    assert schema["additionalProperties"] is False and set(schema["required"]) == set(schema["properties"])
    assert res.output.overall_assessment == "abnormality_detected"
    assert res.usage.reasoning_tokens == 200 and res.usage.estimated_cost_usd == pytest.approx((1000 * 1.25 + 500 * 4.25) / 1e6)


def test_muse_repair_turn(muse_fixture):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        text = "not json at all" if len(calls) == 1 else "```json\n" + json.dumps(muse_fixture) + "\n```"
        return httpx.Response(200, json=_response(text))
    analyzer = MuseSparkAnalyzer(client=MetaModelClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler)))
    res = analyzer.analyze(["data:image/jpeg;base64,AAAA"], {})
    assert len(calls) == 2 and calls[1]["previous_response_id"] == "resp_1"
    assert res.notes and res.output.hypotheses[0].problem_id == "black_pod"


def test_parse_muse_output_strips_fences(muse_fixture):
    out = parse_muse_output("Voici :\n" + json.dumps(muse_fixture))
    assert out.image_assessment.is_cocoa == "yes"


# --------------------------------------------------------------------------- SAM
def _sam_wire(mask: np.ndarray, conf: bool = True) -> str:
    """Build a contract-valid SAM line with Meta's official codec.

    As observed on the live API, the mask payload is LOCAL to the box (data=H,W = box size).
    """
    from meta_sam_parser._mask_codec import _encode_raster  # official encoder (test-only use)

    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    local = mask[y1:y2 + 1, x1:x2 + 1]
    payload = _encode_raster(local.flatten().tolist(), local.shape[1], local.shape[0])
    c = ";c=0.91" if conf else ""
    return (f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}{c}|>"
            f"<|mask;x=0;y=0{c};data={local.shape[0]},{local.shape[1]},{payload}|>")


def test_parse_sam_output_decodes_mask():
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[10:20, 30:50] = 1
    regions, masks, diags = parse_sam_output(_sam_wire(mask), "img1", "brown lesion", (60, 40))
    assert len(regions) == 1 and masks[0].sum() == 200
    r = regions[0]
    assert r.mask_area_fraction == round(200 / 2400, 5)
    assert r.bbox_xyxy[0] == 30 and r.mask_confidence == pytest.approx(0.91)
    assert masks[0].shape == (40, 60) and masks[0][10:20, 30:50].all()  # re-placed at the box position


def test_parse_sam_empty_output_means_no_match():
    regions, masks, _ = parse_sam_output("", "img1", "leaf", (10, 10))
    assert regions == [] and masks == []


def _sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def test_sam_segment_streaming_end_to_end(tmp_path):
    img = Image.new("RGB", (60, 40), (120, 160, 60))
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[5:30, 10:40] = 1
    wire = _sam_wire(mask)
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert body["model"] == "sam-3.1" and body["stream"] is True
        assert body["metadata"]["include_confidence"] == "true"
        parts = body["input"][0]["content"]
        assert parts[0]["type"] == "input_text" and parts[1]["type"] == "input_image"
        events = [{"type": "response.output_text.delta", "delta": wire[:20]},
                  {"type": "response.output_text.delta", "delta": wire[20:]},
                  {"type": "response.completed", "response": {"id": "r", "status": "completed"}}]
        return httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})
    sam = SamSegmenter(client=MetaModelClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler)))
    sam.enabled = True
    res = sam.segment(img, "img1", prompts=["leaf"], artifacts_dir=tmp_path)
    assert res.status == "ok" and len(res.regions) == 1
    assert res.regions[0].crop_path and res.regions[0].overlay_path
    assert len(seen) == 1


def test_sam_never_fabricates_on_failure(tmp_path):
    def handler(request):
        return httpx.Response(404, json={"error": {"message": "model not found"}})
    sam = SamSegmenter(client=MetaModelClient(api_key=FAKE_KEY, transport=httpx.MockTransport(handler), max_retries=0))
    sam.enabled = True
    res = sam.segment(Image.new("RGB", (64, 64)), "img1", prompts=["leaf", "pod"], artifacts_dir=tmp_path)
    assert res.status == "error" and res.regions == []


def test_sam_without_key_is_unavailable():
    sam = SamSegmenter(client=MetaModelClient(api_key=""))
    sam.enabled = True
    assert sam.segment(Image.new("RGB", (64, 64)), "i").status == "unavailable"


def test_proxy_injected_auth_sends_no_authorization_header(monkeypatch):
    monkeypatch.setenv("MAANA_META_PROXY_AUTH", "true")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(401, json={"error": {"message": "Unauthorized"}})
    client = MetaModelClient(transport=httpx.MockTransport(handler), max_retries=0)
    assert client.configured and client.auth_mode == "proxy_injected"
    with pytest.raises(MetaAuthError) as exc:
        client.create_response({"model": "m", "input": "x"})
    assert seen["auth"] is None and "secret réseau" in str(exc.value)


def test_corrupted_keys_are_rejected(muse_fixture):
    """Live run 2026-10-31 returned keys like "hypotheses_foo": they must not pass silently."""
    from pydantic import ValidationError

    bad = dict(muse_fixture)
    bad["hypotheses_foo"] = bad.pop("hypotheses")
    with pytest.raises(ValidationError):
        parse_muse_output(json.dumps(bad))
