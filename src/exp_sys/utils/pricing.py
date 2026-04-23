"""Model pricing for the third-party API we use (lonlie.plus7.plus).

Prices are in **USD per 1M tokens** (input, output). Keep this table in sync
with reference/test_openai_api.py — the source of truth is the proxy's
/api/pricing endpoint; the conversion is `input = model_ratio * 2`,
`output = model_ratio * completion_ratio * 2`.

Last synced: 2026-04-22.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

# Prices in USD per 1,000,000 tokens: {model_id: (input_price, output_price)}
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-3-5-sonnet-20240620":          (3.0,   15.0),
    "claude-3-5-sonnet-20241022":          (3.0,   15.0),
    "claude-3-7-sonnet-20250219":          (3.0,   15.0),
    "claude-3-7-sonnet-20250219-thinking": (3.0,   15.0),
    "claude-4-opus":                       (15.0,  75.0),
    "claude-4-opus-thinking":              (15.0,  75.0),
    "claude-4-sonnet":                     (3.0,   15.0),
    "claude-4-sonnet-thinking":            (3.0,   15.0),
    "claude-haiku-4-5-20251001":           (1.0,    5.0),
    "claude-opus-4-20250514":              (15.0,  75.0),
    "claude-opus-4-20250514-thinking":     (15.0,  75.0),
    "claude-opus-4-5-20251101":            (5.0,   25.0),
    "claude-opus-4-5-20251101-thinking":   (5.0,   25.0),
    "claude-opus-4-6":                     (5.0,   25.0),
    "claude-opus-4-7":                     (5.0,   25.0),
    "claude-sonnet-4-20250514":            (3.0,   15.0),
    "claude-sonnet-4-20250514-thinking":   (75.0, 375.0),
    "claude-sonnet-4-5-20250929":          (3.0,   15.0),
    "claude-sonnet-4-5-20250929-thinking": (3.0,   15.0),
    "claude-sonnet-4-6":                   (3.0,   15.0),
    "claudecode-3-7-sonnet":               (75.0,  75.0),
    "claudecode-3-7-sonnet-thinking":      (75.0,  75.0),
    "claudecode-4-5-haiku":                (75.0,  75.0),
    "claudecode-4-5-opus":                 (5.0,    5.0),
    "claudecode-4-5-opus-thinking":        (5.0,    5.0),
    "claudecode-4-5-sonnet":               (75.0,  75.0),
    "claudecode-4-5-sonnet-thinking":      (75.0,  75.0),
    "claudecode-4-opus":                   (75.0,  75.0),
    "claudecode-4-opus-thinking":          (75.0,  75.0),
    "claudecode-4-sonnet":                 (75.0,  75.0),
    "claudecode-4-sonnet-thinking":        (75.0,  75.0),
    # DeepSeek
    "deepseek-chat":                       (2.0,    3.0),
    "deepseek-reasoner":                   (4.0,   16.0),
    # Google
    "gemini-1.5-flash-exp-0827":           (2.0,    8.0),
    "gemini-1.5-flash-latest":             (2.0,    8.0),
    "gemini-1.5-pro":                      (3.5,   14.0),
    "gemini-1.5-pro-exp-0827":             (3.5,   14.0),
    "gemini-1.5-pro-latest":               (3.5,   14.0),
    "gemini-2.0-flash-exp":                (2.0,    8.0),
    "gemini-2.0-flash-thinking-exp-1219":  (2.0,    8.0),
    "gemini-2.0-pro-exp":                  (75.0, 300.0),
    "gemini-2.5-flash":                    (0.3,  2.499),
    "gemini-2.5-flash-image-preview":      (0.0,    0.0),
    "gemini-2.5-flash-lite":               (75.0, 300.0),
    "gemini-2.5-pro":                      (1.25,  10.0),
    "gemini-3-flash-preview":              (0.5,    3.0),
    "gemini-3-pro-image-preview":          (0.0,    0.0),
    "gemini-3-pro-image-preview-1K":       (75.0, 4500.0),
    "gemini-3-pro-image-preview-2K":       (75.0, 4500.0),
    "gemini-3-pro-image-preview-4K":       (75.0, 4500.0),
    "gemini-3-pro-preview":                (2.0,   12.0),
    "gemini-3-pro-preview-thinking":       (2.0,   12.0),
    "gemini-3.1-flash-image-preview":      (0.0,    0.0),
    "gemini-3.1-flash-preview":            (75.0, 300.0),
    "gemini-3.1-pro-preview":              (2.0,   12.0),
    "gemini-exp-1114":                     (75.0, 300.0),
    "gemini-exp-1121":                     (75.0, 300.0),
    "gemini-exp-1206":                     (75.0, 300.0),
    "gemini-ultra":                        (2.0,    8.0),
    # OpenAI — GPT-3.5
    "gpt-3.5-turbo":                       (0.5,    1.0),
    "gpt-3.5-turbo-0125":                  (0.5,    1.0),
    "gpt-3.5-turbo-0613":                  (1.5,    3.0),
    "gpt-3.5-turbo-1106":                  (1.0,    2.0),
    "gpt-3.5-turbo-16k":                   (3.0,    6.0),
    "gpt-3.5-turbo-16k-0613":              (3.0,    6.0),
    "gpt-3.5-turbo-instruct":              (1.5,    3.0),
    # OpenAI — GPT-4 family
    "gpt-4":                               (30.0,  60.0),
    "gpt-4-0125-preview":                  (10.0,  20.0),
    "gpt-4-0613":                          (30.0,  60.0),
    "gpt-4-32k":                           (60.0, 120.0),
    "gpt-4-32k-0613":                      (60.0, 120.0),
    "gpt-4-turbo":                         (10.0,  30.0),
    "gpt-4-turbo-2024-04-09":              (10.0,  30.0),
    "gpt-4-turbo-preview":                 (10.0,  30.0),
    "gpt-4-vision-preview":                (10.0,  20.0),
    "gpt-4.1":                             (2.0,    8.0),
    "gpt-4.1-2025-04-14":                  (2.0,    8.0),
    "gpt-4.1-mini":                        (0.4,    1.6),
    "gpt-4.5-preview":                     (75.0, 150.0),
    "gpt-4.5-preview-2025-02-27":          (75.0, 150.0),
    # OpenAI — GPT-4o family
    "gpt-4o":                              (2.5,   10.0),
    "gpt-4o-2024-05-13":                   (5.0,   15.0),
    "gpt-4o-2024-08-06":                   (2.5,   10.0),
    "gpt-4o-2024-11-20":                   (2.5,   10.0),
    "gpt-4o-audio-preview":                (2.5,   10.0),
    "gpt-4o-audio-preview-2024-10-01":     (2.5,   10.0),
    "gpt-4o-mini":                         (0.15,   0.6),
    "gpt-4o-mini-2024-07-18":              (0.15,   0.6),
    "gpt-4o-mini-audio-preview-2024-12-17":(75.0, 300.0),
    "gpt-4o-mini-tts":                     (75.0, 1500.0),
    "gpt-4o-realtime-preview":             (5.0,   20.0),
    "gpt-4o-realtime-preview-2024-10-01":  (5.0,   20.0),
    # OpenAI — GPT-5 family
    "gpt-5":                               (1.25,  10.0),
    "gpt-5-chat":                          (1.25,  10.0),
    "gpt-5-mini":                          (0.25,   2.0),
    "gpt-5-thinking":                      (75.0, 600.0),
    "gpt-5-thinking-all":                  (75.0, 150.0),
    "gpt-5.1":                             (1.25,  10.0),
    "gpt-5.1-2025-11-13":                  (1.25,  10.0),
    "gpt-5.1-chat":                        (75.0, 600.0),
    "gpt-5.1-chat-latest":                 (1.25,  10.0),
    "gpt-5.1-codex":                       (1.25,  10.0),
    "gpt-5.2":                             (1.75,  14.0),
    "gpt-5.2-codex":                       (75.0, 600.0),
    "gpt-5.3-codex":                       (1.25,  10.0),
    "gpt-5.4":                             (2.5,   15.0),
    "gpt-image-1":                         (5.0,   40.0),
    # OpenAI — o-series reasoning
    "o1":                                  (15.0,  60.0),
    "o1-mini":                             (1.1,    4.4),
    "o1-mini-2024-09-12":                  (1.1,    4.4),
    "o1-preview":                          (15.0,  60.0),
    "o3":                                  (2.0,    8.0),
    "o3-mini":                             (1.1,    4.4),
    "o3-mini-high":                        (1.1,    4.4),
    "o3-mini-low":                         (1.1,    4.4),
    # xAI
    "grok-beta":                           (75.0,  75.0),
    # Embeddings / legacy
    "text-ada-001":                        (0.4,    0.4),
    "text-babbage-001":                    (0.5,    0.5),
    "text-curie-001":                      (2.0,    2.0),
    "text-davinci-edit-001":               (20.0,  20.0),
    "text-embedding-3-large":              (0.13,   0.13),
    "text-embedding-3-small":              (0.02,   0.02),
    "text-embedding-ada-002":              (0.1,    0.1),
    "text-embedding-v1":                   (0.1,    0.1),
    "text-moderation-latest":              (0.2,    0.2),
    "text-moderation-stable":              (0.2,    0.2),
}


def get_pricing(model: str) -> Optional[Tuple[float, float]]:
    """Return (input, output) prices in USD/1M tokens, or None if unknown."""
    return MODEL_PRICING.get(model)


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute cost in USD. Prompt/completion tokens are absolute counts;
    prices are per 1M tokens. Returns 0.0 if the model is not in the table."""
    p = MODEL_PRICING.get(model)
    if not p:
        return 0.0
    in_price, out_price = p
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000
