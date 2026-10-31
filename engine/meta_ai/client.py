"""Low-level HTTP client for the Meta Model API (OpenAI-compatible Responses API).

Contract (verified 2026-10-31, https://dev.meta.ai/docs/api-reference/responses/schemas):
* ``POST {base_url}/responses`` with ``Authorization: Bearer <MODEL_API_KEY>``
* ``GET  {base_url}/models``
* Non-streaming response: ``output[]`` items; ``type == "message"`` items carry
  ``content[]`` parts of ``type == "output_text"`` (``text``) or ``"refusal"``.
  ``output_text`` is an SDK-only helper and is NOT relied upon here.
* Streaming (``stream: true``): server-sent events whose JSON ``data`` has a
  ``type`` such as ``response.output_text.delta`` / ``response.completed``.

The key is injected per request and never logged; error messages are redacted.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

from engine.config import get_meta_api_key, get_settings, proxy_auth_allowed
from engine.logging_utils import redact

log = logging.getLogger(__name__)


class MetaAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, body: Any = None):
        super().__init__(redact(message))
        self.status_code = status_code
        self.body = body


class MetaNotConfiguredError(MetaAPIError):
    pass


class MetaAuthError(MetaAPIError):
    pass


class MetaRefusalError(MetaAPIError):
    pass


@dataclass
class ResponseResult:
    text: str
    raw: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str | None = None
    status: str | None = None
    latency_ms: float = 0.0
    dropped_params: list[str] = field(default_factory=list)


def extract_output_text(response: dict[str, Any]) -> str:
    """Concatenate ``output_text`` parts of assistant message items."""
    parts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            ptype = part.get("type")
            if ptype == "output_text":
                parts.append(part.get("text") or "")
            elif ptype == "refusal":
                raise MetaRefusalError(f"Model refusal: {part.get('refusal') or part.get('text')}")
    if not parts and isinstance(response.get("output_text"), str):  # tolerate SDK-shaped payloads
        parts.append(response["output_text"])
    return "".join(parts)


class MetaModelClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cfg = get_settings().models["meta_api"]
        # api_key=None -> resolve from env/Kaggle; if still absent, rely on proxy-injected auth when allowed.
        # An explicit api_key="" means "no credential at all" (used by offline tests).
        self._api_key = api_key if api_key is not None else get_meta_api_key()
        self._proxy_auth = api_key is None and not self._api_key and proxy_auth_allowed()
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.timeout_s = timeout_s or cfg.get("timeout_s", 240)
        self.max_retries = cfg.get("max_retries", 2) if max_retries is None else max_retries
        self._transport = transport

    # ------------------------------------------------------------------ utils
    @property
    def configured(self) -> bool:
        return bool(self._api_key) or self._proxy_auth

    @property
    def auth_mode(self) -> str:
        return "env_key" if self._api_key else ("proxy_injected" if self._proxy_auth else "none")

    def _client(self) -> httpx.Client:
        if not self.configured:
            raise MetaNotConfiguredError(
                "Clé Meta absente : configurer un secret réseau (Bearer, hôte api.meta.ai) dans l'environnement cloud, "
                "ou définir MODEL_API_KEY (variable d'environnement, .env local ou secret Kaggle)."
            )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_s, connect=20),
            headers=headers,
            transport=self._transport,
        )

    @staticmethod
    def _raise_for(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
        except Exception:
            body = resp.text[:2000]
        msg = body.get("error", {}).get("message") if isinstance(body, dict) and isinstance(body.get("error"), dict) else str(body)
        if resp.status_code in (401, 403):
            hint = "" if "Authorization" in resp.request.headers else (
                " — aucune clé envoyée par le code : vérifier le secret réseau (type Bearer, hôte api.meta.ai).")
            raise MetaAuthError(f"Authentification Meta refusée ({resp.status_code}): {msg}{hint}", resp.status_code, body)
        raise MetaAPIError(f"Meta API HTTP {resp.status_code}: {msg}", resp.status_code, body)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            try:
                with self._client() as client:
                    resp = client.request(method, path, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                self._raise_for(resp)
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                if attempt >= self.max_retries:
                    raise MetaAPIError(f"Meta API indisponible après {attempt + 1} tentatives : {exc}") from exc
                wait = 2 ** (attempt + 1)
                log.warning("Meta API retry %s in %ss (%s)", attempt + 1, wait, type(exc).__name__)
                time.sleep(wait)
                attempt += 1

    # -------------------------------------------------------------- endpoints
    def list_models(self) -> list[str]:
        resp = self._request("GET", "/models")
        data = resp.json()
        items = data.get("data", data if isinstance(data, list) else [])
        return [m.get("id") for m in items if isinstance(m, dict) and m.get("id")]

    def create_response(self, payload: dict[str, Any], optional_keys: tuple[str, ...] = ()) -> ResponseResult:
        """POST /responses (non-streaming).

        ``optional_keys`` are top-level request fields that may be dropped one by one
        if the API rejects the request with HTTP 400 mentioning them. Each drop is
        recorded in ``dropped_params`` so it is visible in reports (never silent).
        """
        payload = dict(payload)
        dropped: list[str] = []
        t0 = time.perf_counter()
        while True:
            try:
                resp = self._request("POST", "/responses", json=payload)
                break
            except MetaAPIError as exc:
                candidates = [k for k in optional_keys if k in payload]
                if exc.status_code != 400 or not candidates:
                    raise
                body_txt = json.dumps(exc.body) if exc.body is not None else ""
                key = next((k for k in candidates if k in body_txt), candidates[0])
                log.warning("Meta API rejected parameter %r (HTTP 400); retrying without it", key)
                payload.pop(key)
                dropped.append(key)
        data = resp.json()
        if data.get("status") == "failed" or data.get("error"):
            raise MetaAPIError(f"Réponse en échec : {data.get('error')}", resp.status_code, data)
        return ResponseResult(
            text=extract_output_text(data),
            raw=data,
            usage=data.get("usage") or {},
            response_id=data.get("id"),
            status=data.get("status"),
            latency_ms=(time.perf_counter() - t0) * 1000,
            dropped_params=dropped,
        )

    def stream_response(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """POST /responses with ``stream: true``; yield decoded SSE event objects."""
        payload = {**payload, "stream": True}
        with self._client() as client:
            with client.stream("POST", "/responses", json=payload) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    self._raise_for(resp)
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        log.debug("Ignoring non-JSON SSE line")

    def stream_text(self, payload: dict[str, Any]) -> ResponseResult:
        """Run a streaming request and assemble the final output text."""
        t0 = time.perf_counter()
        deltas: list[str] = []
        final: dict[str, Any] | None = None
        for event in self.stream_response(payload):
            etype = event.get("type", "")
            if etype == "response.output_text.delta":
                deltas.append(event.get("delta") or "")
            elif etype in ("response.completed", "response.incomplete"):
                final = event.get("response") or {}
            elif etype in ("response.failed", "error"):
                raise MetaAPIError(f"Flux en échec : {event.get('response', event).get('error', event)}", None, event)
            elif etype == "response.refusal.delta":
                raise MetaRefusalError("Model refusal during stream", None, event)
        text = "".join(deltas)
        if not text and final:
            text = extract_output_text(final)
        return ResponseResult(
            text=text,
            raw=final or {},
            usage=(final or {}).get("usage") or {},
            response_id=(final or {}).get("id"),
            status=(final or {}).get("status"),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
