"""Shared chat completion helper with 429 retry fallback."""

import logging

from openai import AsyncOpenAI, BadRequestError, InternalServerError, RateLimitError

logger = logging.getLogger(__name__)

FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# Conservative free-tier output cap — raise once models/limits are confirmed
MAX_TOKENS = 1024


async def chat_with_fallback(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
) -> tuple[object, str]:
    """Call chat completions, retrying once with the fallback model on 429 or 400.

    Returns (response, model_used).
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )
        if response.choices:
            return response, model
        logger.warning("Empty choices from %s — retrying with fallback %s", model, FALLBACK_MODEL)
    except RateLimitError:
        if model == FALLBACK_MODEL:
            raise
        logger.warning("429 on %s — retrying with fallback %s", model, FALLBACK_MODEL)
    except BadRequestError as exc:
        if model == FALLBACK_MODEL:
            raise
        logger.warning("400 on %s (%s) — retrying with fallback %s", model, exc, FALLBACK_MODEL)
    except InternalServerError as exc:
        if model == FALLBACK_MODEL:
            raise
        logger.warning("503 on %s (%s) — retrying with fallback %s", model, exc, FALLBACK_MODEL)

    response = await client.chat.completions.create(
        model=FALLBACK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    return response, FALLBACK_MODEL
