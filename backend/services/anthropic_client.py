"""
Anthropic API client for the Strat Hub dashboard.

Proxies all calls so API keys never reach the browser bundle. The backend
reads ANTHROPIC_API_KEY from the environment at request time.

Thinking API (confirmed from Anthropic docs 2026-05):

  claude-opus-4-7       → ADAPTIVE ONLY.  thinking:{type:"adaptive"} + output_config.effort
  claude-opus-4-6       → ADAPTIVE recommended (budget_tokens deprecated, will be removed)
  claude-sonnet-4-6     → same as opus-4-6

  claude-opus-4-5-*     → EXTENDED ONLY.  thinking:{type:"enabled", budget_tokens:N} + temperature:1
  claude-sonnet-4-5-*   → same as opus-4-5

CRITICAL: For adaptive models, max_tokens is a HARD CAP on thinking + text combined.
The model self-regulates thinking depth via the effort parameter. If max_tokens is too
low, the model exhausts the budget on thinking alone and returns NO visible text
(stop_reason="max_tokens"). Solution: set max_tokens to the model's actual max_output
limit and let effort control how much thinking occurs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class CountTokensPayloadTooLargeError(Exception):
    """Anthropic count_tokens rejected the request for size (HTTP 413 or HTTP 400 text-byte cap)."""


# Anthropic count_tokens (and related) total text size budget (UTF-8 bytes), per API error messages.
COUNT_TOKENS_TOTAL_TEXT_BYTES_LIMIT = 32_000_000


class MessagesPayloadTooLargeError(Exception):
    """Anthropic /v1/messages rejected the request body (HTTP 413)."""


# ── Constants ─────────────────────────────────────────────────────────────
ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_MODEL = "claude-sonnet-4-6"


def _get_anthropic_api_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return api_key

# ── Model registry ────────────────────────────────────────────────────────
# Exposed to the frontend via GET /catalog → .models
#
# tier:
#   "adaptive"  → thinking:{type:"adaptive"} + output_config.effort (new API)
#   "extended"  → thinking:{type:"enabled", budget_tokens:N} + temperature:1 (older API)
#
# effort_levels: what the UI should allow for this specific model.
#   Adaptive models: off / low / medium / high / max
#   Extended models: off / low / medium / high / max  (mapped to budget_tokens)
#   NOTE: "xhigh" exists only for Opus 4.7 / Mythos; excluded to keep UI simple.
#
# thinking_note: short string shown in the UI hint bar.
SUPPORTED_MODELS: list[dict] = [
    {
        "id":            "claude-opus-4-7",
        "label":         "Opus 4.7",
        "tier":          "adaptive",
        "context_window": 1_000_000,
        "max_output":    128_000,
        "effort_levels": ["off", "low", "medium", "high", "max"],
        "thinking_note": "Adaptive — always reasons at ≥high by default; effort tunes depth.",
    },
    {
        "id":            "claude-opus-4-6",
        "label":         "Opus 4.6",
        "tier":          "adaptive",
        "context_window": 1_000_000,
        "max_output":    128_000,
        "effort_levels": ["off", "low", "medium", "high", "max"],
        "thinking_note": "Adaptive — Claude decides when to think; effort guides depth.",
    },
    {
        "id":            "claude-sonnet-4-6",
        "label":         "Sonnet 4.6",
        "tier":          "adaptive",
        "context_window": 1_000_000,
        "max_output":    64_000,
        "effort_levels": ["off", "low", "medium", "high", "max"],
        "thinking_note": "Adaptive — fast + reasoning; best speed/quality balance.",
    },
    {
        "id":            "claude-opus-4-5-20251101",
        "label":         "Opus 4.5",
        "tier":          "extended",
        "context_window": 200_000,
        "max_output":    64_000,
        "effort_levels": ["off", "low", "medium", "high", "max"],
        "thinking_note": "Extended — reasoning budget: off=0 · low=2k · medium=8k · high=24k · max=48k tokens.",
    },
    {
        "id":            "claude-sonnet-4-5-20250929",
        "label":         "Sonnet 4.5",
        "tier":          "extended",
        "context_window": 200_000,
        "max_output":    64_000,
        "effort_levels": ["off", "low", "medium", "high", "max"],
        "thinking_note": "Extended — reasoning budget: off=0 · low=2k · medium=8k · high=24k · max=48k tokens.",
    },
    {
        "id":            "claude-haiku-3-5-20241022",
        "label":         "Haiku 3.5",
        "tier":          "basic",
        "context_window": 200_000,
        "max_output":    8_192,
        "effort_levels": ["off"],
        "thinking_note": "No thinking — fastest and most cost-effective; best for simple analyses.",
    },
]

# IDs that use adaptive thinking (output_config.effort, no budget_tokens).
_ADAPTIVE_IDS: frozenset[str] = frozenset(
    m["id"] for m in SUPPORTED_MODELS if m["tier"] == "adaptive"
)

# IDs that use extended thinking (budget_tokens).
_EXTENDED_IDS: frozenset[str] = frozenset(
    m["id"] for m in SUPPORTED_MODELS if m["tier"] == "extended"
)

# effort label → budget_tokens for extended-thinking models.
_EFFORT_BUDGETS: dict[str, int] = {
    "off":    0,
    "low":    2_000,
    "medium": 8_000,
    "high":   24_000,
    "max":    48_000,
}

# All effort options shown in the UI (union across all model tiers).
EFFORT_OPTIONS: list[str] = ["off", "low", "medium", "high", "max"]

DEFAULT_EFFORT = "high"


def _is_adaptive(model: str) -> bool:
    """True for models that use adaptive thinking (output_config.effort)."""
    return model in _ADAPTIVE_IDS


def _model_max_output(model: str) -> int:
    """Return the model's actual max output token limit."""
    for m in SUPPORTED_MODELS:
        if m["id"] == model:
            return int(m.get("max_output", 64_000))
    return 64_000


def _messages_count_body(
    *,
    system: str,
    user: str,
    model: str | None,
    effort: str,
) -> dict[str, Any]:
    """Request body aligned with Messages API usage (for /messages + count_tokens)."""
    chosen = (model or DEFAULT_MODEL).strip()
    eff = (effort or "medium").lower()
    adaptive = _is_adaptive(chosen)
    extended = chosen in _EXTENDED_IDS

    body: dict[str, Any] = {
        "model": chosen,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if eff == "off" or (not adaptive and not extended):
        pass  # vanilla text-in
    elif adaptive:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": eff}
    else:
        budget = _EFFORT_BUDGETS.get(eff, _EFFORT_BUDGETS["medium"])
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        body["temperature"] = 1
    return body


async def count_message_input_tokens(
    *,
    system: str,
    user: str,
    model: str | None,
    effort: str = DEFAULT_EFFORT,
    timeout_s: float = 120.0,
) -> int:
    """Anthropic tokenizer count via POST /v1/messages/count_tokens (same shaping as Messages)."""
    body = _messages_count_body(system=system, user=user, model=model, effort=effort)
    headers = {
        "x-api-key":           _get_anthropic_api_key(),
        "anthropic-version":   ANTHROPIC_VERSION,
        "content-type":        "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(COUNT_TOKENS_URL, json=body, headers=headers)

    if resp.status_code == 413:
        logger.error("anthropic count_tokens 413: %s", resp.text[:500])
        raise CountTokensPayloadTooLargeError(resp.text[:500])
    if resp.status_code == 400:
        low = resp.text.lower()
        if "too many total text bytes" in low or "request exceeds the maximum size" in low:
            logger.error("anthropic count_tokens 400 (text size): %s", resp.text[:500])
            raise CountTokensPayloadTooLargeError(resp.text[:500])
    if resp.status_code != 200:
        logger.error("anthropic count_tokens %s: %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Anthropic count_tokens {resp.status_code}: {resp.text[:350]}")

    data = resp.json()
    return int(data.get("input_tokens", 0))


async def call_claude(
    *,
    system: str,
    user: str,
    model: str | None = None,
    effort: str = DEFAULT_EFFORT,
    max_tokens: int = 4_000,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Call Anthropic /v1/messages and return a normalised result dict.

    Returns
    -------
    {
        "text":       str,
        "thinking":   str | None,
        "tokens_in":  int,
        "tokens_out": int,
        "model":      str,
        "stop_reason": str,
    }
    """
    chosen = (model or DEFAULT_MODEL).strip()
    eff     = (effort or "medium").lower()
    adaptive = _is_adaptive(chosen)
    extended = chosen in _EXTENDED_IDS

    body = _messages_count_body(system=system, user=user, model=model, effort=effort)

    if eff == "off" or (not adaptive and not extended):
        body["max_tokens"]  = max_tokens
        body["temperature"] = 0.4
    elif adaptive:
        # max_tokens is a HARD CAP on thinking + visible text combined.
        # The model self-regulates thinking via the effort param — we just need
        # to give it enough headroom so it doesn't exhaust the budget on thinking
        # alone and return no text. Use the model's actual max_output ceiling.
        model_max_output = _model_max_output(chosen)
        body["max_tokens"] = model_max_output
    else:
        budget = _EFFORT_BUDGETS.get(eff, _EFFORT_BUDGETS["medium"])
        body["max_tokens"]  = max(max_tokens, budget + max_tokens)

    headers = {
        "x-api-key":           _get_anthropic_api_key(),
        "anthropic-version":   ANTHROPIC_VERSION,
        "content-type":        "application/json",
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            resp = await client.post(ANTHROPIC_URL, json=body, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("anthropic transport error: %s", exc)
            raise

    if resp.status_code == 413:
        logger.error("anthropic messages 413: %s", resp.text[:500])
        raise MessagesPayloadTooLargeError(resp.text[:500])
    if resp.status_code == 400:
        low = resp.text.lower()
        if "too many total text bytes" in low:
            logger.error("anthropic messages 400 (text size): %s", resp.text[:500])
            raise MessagesPayloadTooLargeError(resp.text[:500])
    if resp.status_code != 200:
        logger.error("anthropic %s: %s", resp.status_code, resp.text[:500])
        raise RuntimeError(f"Anthropic API {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []
    for block in data.get("content") or []:
        btype = block.get("type")
        if btype == "text":
            text_chunks.append(block.get("text", ""))
        elif btype == "thinking":
            thinking_chunks.append(block.get("thinking", ""))

    stop_reason = data.get("stop_reason", "")
    usage = data.get("usage") or {}
    final_text = "".join(text_chunks).strip()

    if not final_text and thinking_chunks:
        logger.warning(
            "Claude returned thinking (%d chars) but no text output. "
            "stop_reason=%s, max_tokens=%s, output_tokens=%s. "
            "This usually means max_tokens was exhausted by thinking.",
            sum(len(t) for t in thinking_chunks),
            stop_reason,
            body.get("max_tokens"),
            usage.get("output_tokens"),
        )

    return {
        "text":        final_text,
        "thinking":    "\n".join(t for t in thinking_chunks if t).strip() or None,
        "tokens_in":   int(usage.get("input_tokens",  0)),
        "tokens_out":  int(usage.get("output_tokens", 0)),
        "model":       data.get("model", chosen),
        "stop_reason": stop_reason,
    }
