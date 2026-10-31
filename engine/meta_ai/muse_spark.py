"""Muse Spark adapter: multimodal analysis with structured JSON output.

Request (Responses API, verified against https://dev.meta.ai/docs/image-understanding,
/docs/structured-output and /docs/reasoning):

    {"model": "muse-spark-1.3",
     "instructions": "...",
     "input": [{"type": "message", "role": "user", "content": [
         {"type": "input_text", "text": "..."},
         {"type": "input_image", "image_url": "data:image/jpeg;base64,..."}]}],
     "reasoning": {"effort": "medium"},
     "text": {"format": {"type": "json_schema", "name": "...", "schema": {...}, "strict": false}},
     "max_output_tokens": 8000}

The model output is ALWAYS re-validated locally with Pydantic; one repair turn is
attempted (``previous_response_id``) if the JSON is invalid.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from engine.config import get_settings
from engine.meta_ai.client import MetaAPIError, MetaModelClient, ResponseResult
from engine.meta_ai.prompts import PROMPT_VERSION, REPAIR_PROMPT, system_instructions, user_preamble
from engine.schemas import ModelUsage, MuseAnalysisOutput

log = logging.getLogger(__name__)

SCHEMA_NAME = "maana_terra_cocoa_analysis"


class MuseOutputError(MetaAPIError):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


@dataclass
class MuseCallResult:
    output: MuseAnalysisOutput
    raw_text: str
    usage: ModelUsage
    latency_ms: float
    response_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION


def _strip_to_json(text: str) -> str:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def strict_json_schema(model: type) -> dict[str, Any]:
    """Pydantic schema -> Meta strict-mode subset: every object has additionalProperties=false
    and lists ALL its properties in ``required``; defaults/titles removed."""
    schema = model.model_json_schema()

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def parse_muse_output(text: str) -> MuseAnalysisOutput:
    return MuseAnalysisOutput.model_validate(json.loads(_strip_to_json(text)))


def estimate_cost(model: str, usage: dict[str, Any]) -> float | None:
    price = get_settings().models.get("pricing", {}).get(model)
    if not price or "input_per_mtok" not in price:
        return None
    inp = int(usage.get("input_tokens") or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    return round(
        ((inp - cached) * price["input_per_mtok"] + cached * price.get("cached_input_per_mtok", price["input_per_mtok"])
         + out * price["output_per_mtok"]) / 1e6,
        6,
    )


class MuseSparkAnalyzer:
    def __init__(self, client: MetaModelClient | None = None, model: str | None = None) -> None:
        s = get_settings()
        self.cfg = s.models["muse_spark"]
        self.client = client or MetaModelClient()
        self.model = model or s.muse_model
        self.language = s.report_language

    def build_payload(self, image_data_urls: list[str], context: dict[str, Any], has_rois: bool) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_preamble(len(image_data_urls), has_rois)},
            {"type": "input_text", "text": "CONTEXT:\n" + json.dumps(context, ensure_ascii=False, default=str)},
        ]
        content += [{"type": "input_image", "image_url": url} for url in image_data_urls]
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": system_instructions(self.language),
            "input": [{"type": "message", "role": "user", "content": content}],
            "max_output_tokens": self.cfg.get("max_output_tokens", 8000),
        }
        if self.cfg.get("reasoning_effort"):
            payload["reasoning"] = {"effort": self.cfg["reasoning_effort"]}
        if self.cfg.get("temperature") is not None:
            payload["temperature"] = self.cfg["temperature"]
        if self.cfg.get("structured_output", True):
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": SCHEMA_NAME,
                    "schema": strict_json_schema(MuseAnalysisOutput) if self.cfg.get("strict_schema", True)
                    else MuseAnalysisOutput.model_json_schema(),
                    "strict": bool(self.cfg.get("strict_schema", True)),
                }
            }
        return payload

    def _usage(self, results: list[ResponseResult], n_images: int) -> ModelUsage:
        agg: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0}}
        reasoning = 0
        for r in results:
            u = r.usage or {}
            agg["input_tokens"] += int(u.get("input_tokens") or 0)
            agg["output_tokens"] += int(u.get("output_tokens") or 0)
            agg["input_tokens_details"]["cached_tokens"] += int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
            reasoning += int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
        return ModelUsage(
            model=self.model,
            input_tokens=agg["input_tokens"],
            output_tokens=agg["output_tokens"],
            reasoning_tokens=reasoning,
            cached_tokens=agg["input_tokens_details"]["cached_tokens"],
            images=n_images,
            estimated_cost_usd=estimate_cost(self.model, agg),
            response_id=results[-1].response_id if results else None,
        )

    def analyze(self, image_data_urls: list[str], context: dict[str, Any], has_rois: bool = False) -> MuseCallResult:
        if not image_data_urls:
            raise ValueError("At least one image is required")
        payload = self.build_payload(image_data_urls, context, has_rois)
        first = self.client.create_response(payload, optional_keys=("text", "reasoning"))
        results = [first]
        notes = [f"Paramètre rejeté par l'API et retiré : {k}" for k in first.dropped_params]
        try:
            output = parse_muse_output(first.text)
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning("Muse Spark output invalid, attempting one repair turn: %s", type(exc).__name__)
            notes.append("Sortie JSON initiale invalide ; un tour de réparation a été effectué.")
            repair: dict[str, Any] = {
                "model": self.model,
                "instructions": system_instructions(self.language),
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": REPAIR_PROMPT.format(errors=str(exc)[:3000])}]}],
                "max_output_tokens": self.cfg.get("max_output_tokens", 8000),
            }
            if first.response_id:
                repair["previous_response_id"] = first.response_id
            if "text" in payload and "text" not in first.dropped_params:
                repair["text"] = payload["text"]
            second = self.client.create_response(repair, optional_keys=("text",))
            results.append(second)
            try:
                output = parse_muse_output(second.text)
            except (json.JSONDecodeError, ValidationError) as exc2:
                raise MuseOutputError(f"Sortie Muse Spark non conforme au schéma : {exc2}", second.text) from exc2
        return MuseCallResult(
            output=output,
            raw_text=results[-1].text,
            usage=self._usage(results, len(image_data_urls)),
            latency_ms=sum(r.latency_ms for r in results),
            response_ids=[r.response_id for r in results if r.response_id],
            notes=notes,
        )
