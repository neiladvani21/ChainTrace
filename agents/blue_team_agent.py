"""Blue Team agent — challenges attack steps with environment-specific controls.

Model: qwen/qwen3-80b-a3b:free via OpenRouter.
"""

import json
import logging
import time
from typing import Optional

from openai import AsyncOpenAI

from agents.base import chat_with_fallback

logger = logging.getLogger(__name__)

MODEL = "nousresearch/hermes-3-llama-3.1-405b:free"

SYSTEM_PROMPT = """You are a senior defensive security engineer who knows this \
environment intimately. Challenge every attack step the red team proposes. For each \
step, either name a specific control that blocks it (WAF rule, egress filter, network \
segmentation, patch status) or concede that the gap exists. Only reference controls \
that exist in the environment description provided. Do not invent defenses. Be \
precise and honest."""


class BlueTeamAgent:
    """Defensive agent that challenges each red team attack step."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client
        self.model = MODEL
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def run(
        self,
        cve_data: list[dict],
        environment: dict,
        debate_history: list[dict],
    ) -> tuple[str, int, int]:
        """Generate a defensive challenge to the latest red team proposal.

        Returns (response_text, input_tokens, output_tokens).
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        context = _build_context(cve_data, environment)
        messages.append({"role": "user", "content": context})

        for turn in debate_history:
            role = "user" if turn["role"] == "red" else "assistant"
            messages.append({"role": role, "content": turn["content"]})

        messages.append(
            {
                "role": "user",
                "content": (
                    "The red team has proposed the attack chain above. For each step, "
                    "state whether a specific control in this environment blocks it, "
                    "or explicitly concede the gap. Do not invent defenses that are "
                    "not present in the environment description."
                ),
            }
        )

        start = time.monotonic()
        response, used_model = await chat_with_fallback(self.client, self.model, messages, temperature=0.2)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        msg = response.choices[0].message
        # Some reasoning models (Qwen, DeepSeek) return the actual answer in
        # reasoning_content when content is empty — fall back to that field.
        text = (msg.content or "").strip()
        if not text:
            text = getattr(msg, "reasoning_content", "") or ""
            text = text.strip()
        if not text:
            text = "[Blue Team response unavailable — model returned empty content]"
            logger.warning("BlueTeam got empty content from %s", used_model)
        else:
            logger.info("BlueTeam (%s): %d chars", used_model, len(text))

        in_tok = response.usage.prompt_tokens if response.usage else 0
        out_tok = response.usage.completion_tokens if response.usage else 0
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        logger.debug("BlueTeam response: %d ms, %d+%d tokens", elapsed_ms, in_tok, out_tok)
        return text, in_tok, out_tok


def _build_context(cve_data: list[dict], environment: dict) -> str:
    cve_block = "\n\n".join(
        f"CVE: {c['id']}\n"
        f"Severity: {c.get('severity', 'UNKNOWN')} (score {c.get('score', 0)})\n"
        f"Description: {c.get('description', '')}\n"
        f"CVSS Vector: {c.get('vector', '')}\n"
        f"In CISA KEV: {c.get('in_kev', False)}\n"
        f"ATT&CK Techniques: {', '.join(t['id'] + ' ' + t['name'] for t in c.get('techniques', []))}"
        for c in cve_data
    )
    env_block = json.dumps(environment, indent=2)
    return (
        f"## CVE Intelligence\n{cve_block}\n\n"
        f"## Environment (only reference controls listed here)\n```json\n{env_block}\n```"
    )
