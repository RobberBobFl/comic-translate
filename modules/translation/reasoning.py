import json
import urllib.parse

import requests

# Capability cache keyed by (api_base_url, model) -> supports "off".
# Determined once per model/endpoint and reused for the app session.
_CAP_CACHE: dict[tuple[str, str], bool] = {}

CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 10


def is_ollama_endpoint(api_base_url: str) -> bool:
    """Return True when the URL points at a local Ollama server."""
    parsed = urllib.parse.urlparse(api_base_url or "")
    hostname = (parsed.hostname or "").lower()
    if parsed.port == 11434:
        return True
    if "ollama" in hostname:
        return True
    return False


def _lmstudio_models_url(api_base_url: str) -> str:
    """Derive the LM Studio Native REST models endpoint from a base URL.

    http://host:port/v1[/chat/completions] -> http://host:port/api/v1/models
    """
    parsed = urllib.parse.urlparse(api_base_url or "")
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc or "localhost:1234"
    return f"{scheme}://{netloc}/api/v1/models"


def _supports_off(api_base_url: str, model: str) -> bool:
    """Ask LM Studio whether the model allows turning reasoning off.

    Only acts on a confirmed LM Studio ``/api/v1/models`` response; any error,
    missing model, or missing capability is treated as "cannot disable".
    """
    try:
        url = _lmstudio_models_url(api_base_url)
        resp = requests.get(
            url,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return False

    models = (data or {}).get("models") or []
    for entry in models:
        if entry.get("key") != model:
            continue
        reasoning = (entry.get("capabilities") or {}).get("reasoning")
        if not reasoning:
            return False
        allowed = reasoning.get("allowed_options") or []
        return "off" in allowed
    return False


def get_reasoning_off_params(
    api_base_url: str, model: str, is_ollama: bool = False
) -> dict:
    """Return the params that disable reasoning for a local model.

    Ollama: native ``{"think": False}`` (reliable, no capability lookup).

    LM Studio: when ``capabilities.reasoning.allowed_options`` contains
    ``"off"``, the model supports disabling reasoning, so we request it via the
    OpenAI-compatible ``/v1/chat/completions`` parameter
    ``reasoning_effort: "none"``. Note the mismatch: the capability flag is
    ``"off"`` but the API value LM Studio (and OpenRouter) accepts is
    ``"none"`` -- sending ``"off"`` returns HTTP 400.

    When ``"off"`` is absent or the capability is unknown, return ``{}`` so the
    request is sent unchanged. Any failure (network, missing model, unsupported
    parameter) fails safe to ``{}`` -- we never retry with other variants.
    """
    if is_ollama:
        return {"think": False}

    cache_key = (api_base_url, model)
    if cache_key in _CAP_CACHE:
        return {"reasoning_effort": "none"} if _CAP_CACHE[cache_key] else {}

    supported = _supports_off(api_base_url, model)
    _CAP_CACHE[cache_key] = supported
    return {"reasoning_effort": "none"} if supported else {}
