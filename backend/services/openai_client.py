"""
OpenAI API client for Strat Hub.

Implements a strict "GPT-5.5 Instant" mode:
- Uses Chat Completions API
- Forces reasoning_effort="none" every request
- Keeps UI/API model id stable while mapping to OpenAI model alias
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


for env_path in (
    Path(__file__).resolve().parents[2] / ".env",
    Path(__file__).resolve().parents[3] / ".env",
):
    load_dotenv(env_path, override=False)


def _get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return api_key


SUPPORTED_MODELS: list[dict[str, Any]] = [
    {
        # UI id consumed by frontend; mapped to api_model below.
        "id":            "gpt-5.5-instant",
        "api_model":     "gpt-5.5",
        "label":         "GPT-5.5 Instant",
        "tier":          "instant",
        "context_window": 1_050_000,
        "max_output":    128_000,
        "effort_levels": ["off"],
        "thinking_note": "Instant mode — reasoning is forced OFF for lowest latency.",
        "provider":      "openai",
    },
]

_MODEL_BY_ID: dict[str, dict[str, Any]] = {m["id"]: m for m in SUPPORTED_MODELS}


async def call_openai(
    *,
    system: str,
    user: str,
    model: str = "gpt-5.5-instant",
    effort: str = "medium",
    max_tokens: int = 16_000,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Call OpenAI /v1/chat/completions and return a normalised result dict.

    Returns same shape as anthropic_client.call_claude:
    {
        "text":        str,
        "thinking":    str | None,   (reasoning_content if surfaced)
        "tokens_in":   int,
        "tokens_out":  int,
        "model":       str,
        "stop_reason": str,
    }
    """
    model_cfg = _MODEL_BY_ID.get(model)
    api_model = (model_cfg or {}).get("api_model", model)
    # Instant mode is explicit: never let OpenAI default to medium reasoning.
    reasoning_effort = "none"

    body: dict[str, Any] = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }

    headers = {
        "Authorization": f"Bearer {_get_openai_api_key()}",
        "Content-Type":  "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            resp = await client.post(OPENAI_CHAT_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("openai transport error: %s", exc)
            raise

    if resp.status_code != 200:
        logger.error("openai %s: %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"OpenAI API {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage", {})

    content = message.get("content")
    text = ""
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "output_text"):
                parts.append(str(part.get("text", "")))
        text = "".join(parts)

    reasoning = message.get("reasoning_content") or None
    finish_reason = choice.get("finish_reason", "")

    return {
        "text":        text.strip(),
        "thinking":    reasoning.strip() if reasoning else None,
        "tokens_in":   int(usage.get("prompt_tokens", 0)),
        "tokens_out":  int(usage.get("completion_tokens", 0)),
        "model":       data.get("model", api_model),
        "stop_reason": finish_reason,
    }
