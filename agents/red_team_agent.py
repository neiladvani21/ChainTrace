"""Red Team agent — proposes and defends offensive attack chains.

Model: openai/gpt-oss-120b:free via OpenRouter.
"""

import json
import logging
import time
from typing import Optional

from openai import AsyncOpenAI

from agents.base import chat_with_fallback

logger = logging.getLogger(__name__)

MODEL = "moonshotai/kimi-k2.6:free"

SYSTEM_PROMPT = """You are an elite offensive security researcher. Your job is to \
propose realistic attack chains across multiple CVEs in a given environment. Be \
specific about each step. Cite the CVE that enables each step. Assume you can bypass \
defenses unless the blue team proves otherwise with a specific control. When \
challenged, respond with known bypass techniques or concede only if the defense is \
truly airtight. Always ground your reasoning in the actual CVE data provided to you."""


class RedTeamAgent:
    """Offensive agent that proposes and iterates attack chains."""

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client
        self.model = MODEL
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    async def run(
        self,
        cve_data: list[dict],
        environment: dict,
        debate_history: Optional[list[dict]] = None,
    ) -> tuple[str, int, int]:
        """Generate an offensive proposal or rebuttal.

        Returns (response_text, input_tokens, output_tokens).
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Inject real CVE data so the model cannot hallucinate details
        context = _build_context(cve_data, environment)
        messages.append({"role": "user", "content": context})

        if debate_history:
            for turn in debate_history:
                role = "assistant" if turn["role"] == "red" else "user"
                messages.append({"role": role, "content": turn["content"]})
            # Prompt a rebuttal when the last turn was blue
            if debate_history[-1]["role"] == "blue":
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The blue team has responded above. Counter their defensive "
                            "claims, exploit any gaps they conceded, and refine your "
                            "attack chain accordingly."
                        ),
                    }
                )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Propose a detailed, step-by-step attack chain using the CVEs "
                        "above in the described environment. Be specific about how each "
                        "CVE is used and what the attacker gains at each step."
                    ),
                }
            )

        start = time.monotonic()
        response, used_model = await chat_with_fallback(self.client, self.model, messages, temperature=0.3)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        msg = response.choices[0].message
        text = (msg.content or "").strip()
        if not text:
            text = getattr(msg, "reasoning_content", "") or ""
            text = text.strip()
        if not text:
            logger.warning("RedTeam got empty content from %s", used_model)
        in_tok = response.usage.prompt_tokens if response.usage else 0
        out_tok = response.usage.completion_tokens if response.usage else 0
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        logger.debug("RedTeam response: %d ms, %d+%d tokens", elapsed_ms, in_tok, out_tok)
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
        f"## Target Environment\n```json\n{env_block}\n```"
    )
