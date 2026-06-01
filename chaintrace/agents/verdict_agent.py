"""Verdict agent — synthesizes the full debate into a structured JSON report.

Model: deepseek/deepseek-v4-flash:free via OpenRouter.
DeepSeek is fast and reliable at structured JSON output.
"""

import json
import logging
import re
import time

from openai import AsyncOpenAI

from agents.base import chat_with_fallback
from schema import (
    AttackChainStep,
    ConfidenceBreakdown,
    RemediationItem,
    Telemetry,
    VerdictOutput,
)

logger = logging.getLogger(__name__)

MODEL = "deepseek/deepseek-v4-flash:free"

SYSTEM_PROMPT = """You are a chief security officer synthesizing a red team vs blue \
team debate. Based on the full debate transcript, produce a structured verdict. Be \
specific about which attack steps succeeded, which were blocked, and why. Produce a \
confidence score with a breakdown across four dimensions. Produce a prioritized \
remediation plan based on what the debate revealed.

You MUST respond with ONLY a valid JSON object matching this exact schema:
{
  "attack_chain_viable": bool,
  "confidence_score": float (0.0-1.0),
  "confidence_breakdown": {
    "exploit_maturity": "string describing maturity level",
    "environment_exposure": "string describing exposure level",
    "defender_visibility": "string describing visibility level",
    "patch_availability": "string describing patch status"
  },
  "attack_chain_steps": [
    {
      "step_number": int,
      "cve_id": "CVE-XXXX-XXXXX",
      "action": "what the attacker does at this step",
      "blocked": bool,
      "blocking_control": "control that blocked it or null"
    }
  ],
  "critical_gaps": ["gap1", "gap2"],
  "remediation": [
    {
      "priority": "critical|high|medium",
      "action": "specific fix action",
      "cve_id": "CVE-XXXX-XXXXX",
      "effort": "patch|config_change|network_rule"
    }
  ]

STRICT RULES — violation will break the system:
- cve_id MUST always be a real CVE ID string like "CVE-2021-44228". NEVER null, NEVER omit it.
- Every attack_chain_step MUST reference one of the CVEs provided in the input.
- Every remediation item MUST reference one of the CVEs provided in the input.
- blocking_control is the ONLY field allowed to be null.
}

Do not include any text before or after the JSON."""


class VerdictAgent:
    """Synthesis agent that produces a structured VerdictOutput from the debate."""

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
        telemetry: Telemetry,
    ) -> VerdictOutput:
        """Synthesize the debate into a VerdictOutput.

        Telemetry is passed in so the verdict agent can embed the final
        token counts (including its own) after parsing.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        transcript = "\n\n".join(
            f"[{turn['role'].upper()} - Round {i // 2 + 1}]\n{turn['content']}"
            for i, turn in enumerate(debate_history)
        )

        cve_summary = ", ".join(
            f"{c['id']} ({c.get('severity', 'UNKNOWN')} {c.get('score', 0)})"
            for c in cve_data
        )
        env_block = json.dumps(environment, indent=2)

        user_msg = (
            f"## CVEs Analyzed\n{cve_summary}\n\n"
            f"## Environment\n```json\n{env_block}\n```\n\n"
            f"## Full Debate Transcript\n{transcript}\n\n"
            "Produce the structured JSON verdict now."
        )
        messages.append({"role": "user", "content": user_msg})

        start = time.monotonic()
        response, used_model = await chat_with_fallback(self.client, self.model, messages, temperature=0.1)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        msg = response.choices[0].message
        raw = (msg.content or "").strip()
        if not raw:
            raw = getattr(msg, "reasoning_content", "") or ""
            raw = raw.strip()
        if not raw:
            logger.warning("Verdict got empty content from %s", used_model)
        text = _extract_json(raw or "{}")
        in_tok = response.usage.prompt_tokens if response.usage else 0
        out_tok = response.usage.completion_tokens if response.usage else 0
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok

        logger.debug("Verdict response: %d ms, %d+%d tokens", elapsed_ms, in_tok, out_tok)

        # Merge verdict agent tokens into the telemetry object
        telemetry.verdict_tokens = in_tok + out_tok
        telemetry.total_latency_ms += elapsed_ms

        return _parse_verdict(text, telemetry)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model response.

    Some models wrap their JSON in markdown code fences or add prose before/after.
    We strip fences first, then fall back to finding the outermost { } block.
    """
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)

    # Find the outermost braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text


def _parse_verdict(raw_json: str, telemetry: Telemetry) -> VerdictOutput:
    """Parse the LLM JSON response into a validated VerdictOutput."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse verdict JSON: %s", exc)
        data = {}

    breakdown = data.get("confidence_breakdown", {})
    steps = []
    for i, s in enumerate(data.get("attack_chain_steps", [])):
        cve_id = s.get("cve_id")
        if not cve_id or cve_id is None:
            logger.error("Verdict model returned null cve_id for step %d — model prompt violation", i + 1)
            cve_id = "UNKNOWN"
        steps.append(AttackChainStep(
            step_number=s.get("step_number", i + 1),
            cve_id=cve_id,
            action=s.get("action") or "",
            blocked=bool(s.get("blocked", False)),
            blocking_control=s.get("blocking_control") or None,
        ))
    remediation = [
        RemediationItem(
            priority=r.get("priority") or "medium",
            action=r.get("action") or "",
            cve_id=r.get("cve_id") or "UNKNOWN",
            effort=r.get("effort") or "patch",
        )
        for r in data.get("remediation", [])
    ]

    return VerdictOutput(
        attack_chain_viable=bool(data.get("attack_chain_viable", False)),
        confidence_score=float(data.get("confidence_score", 0.0)),
        confidence_breakdown=ConfidenceBreakdown(
            exploit_maturity=breakdown.get("exploit_maturity", "unknown"),
            environment_exposure=breakdown.get("environment_exposure", "unknown"),
            defender_visibility=breakdown.get("defender_visibility", "unknown"),
            patch_availability=breakdown.get("patch_availability", "unknown"),
        ),
        attack_chain_steps=steps,
        critical_gaps=data.get("critical_gaps", []),
        remediation=remediation,
        telemetry=telemetry,
    )
