from __future__ import annotations

import platform
from typing import Any

import httpx
import ollama

DEFAULT_HOST = "http://localhost:11434"

_cache: dict[str, dict[str, str]] = {}


def _runtime_version(host: str) -> str:
    try:
        return str(httpx.get(f"{host}/api/version", timeout=5).json()["version"])
    except Exception:
        return "unknown"


def _digests(host: str) -> dict[str, str]:
    try:
        return {m.model: (m.digest or "")[:12] for m in ollama.Client(host=host).list().models}
    except Exception:
        return {}


def describe(*model_names: str, host: str = DEFAULT_HOST) -> dict[str, Any]:
    """What produced a run, beyond the model name.

    A run is only reproducible against the same weights on the same runtime. The
    model name alone does not pin either - two runs a fortnight apart matched on
    name, seed and temperature and still produced different drafts.
    """
    if host not in _cache:
        _cache[host] = {"ollama": _runtime_version(host), **_digests(host)}
    known = _cache[host]

    def digest(name: str) -> str:
        """Ollama reports an untagged pull as ':latest'; callers use the bare name."""
        return known.get(name) or known.get(f"{name}:latest", "unknown")

    return {
        "ollama": known.get("ollama", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "models": {name: digest(name) for name in model_names},
    }
