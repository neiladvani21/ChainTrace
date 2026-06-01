"""Debate orchestrator — drives the Red vs Blue debate loop and calls the Verdict agent.

Architecture note: we use a custom loop rather than ADK's built-in sequential/parallel
runners because the debate is stateful and conditional — each round depends on the
previous, and early termination on consensus avoids wasted LLM calls.

All three agents run via OpenRouter using free-tier models:
  Red Team  — openai/gpt-oss-120b:free         (offensive, creative attack reasoning)
  Blue Team — qwen/qwen3-80b-a3b:free          (defensive, precise reasoning)
  Verdict   — deepseek/deepseek-v4-flash:free  (structured JSON synthesis)
"""

import asyncio
import logging
import os
import time

from openai import AsyncOpenAI

from agents.blue_team_agent import BlueTeamAgent
from agents.red_team_agent import RedTeamAgent
from agents.verdict_agent import VerdictAgent
from schema import DebateResult, Telemetry
from tools.cisa import check_kev, load_kev_catalogue
from tools.mitre import get_techniques_for_cve, load_attack_data
from tools.nvd import fetch_cve

logger = logging.getLogger(__name__)

# All models are free-tier on OpenRouter — cost is effectively $0
_INPUT_PRICE_PER_M = 0.0
_OUTPUT_PRICE_PER_M = 0.0

MAX_ROUNDS = 2

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _make_openrouter_client() -> AsyncOpenAI:
    """Create an OpenAI-compatible client pointed at OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
    return AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


# ── CVE data enrichment ────────────────────────────────────────────────────────

async def fetch_all_cve_data(cve_ids: list[str]) -> list[dict]:
    """Fetch NVD + CISA KEV + MITRE ATT&CK data for every CVE.

    nvdlib enforces a 6-second rate-limit between requests without an API key,
    so we fetch sequentially to respect that constraint.
    """
    api_key = os.getenv("NVD_API_KEY")
    results: list[dict] = []

    for cve_id in cve_ids:
        nvd_data = await fetch_cve(cve_id, api_key)
        kev_data = check_kev(cve_id)
        mitre_data = get_techniques_for_cve(
            cve_id,
            nvd_data.get("description", ""),
            nvd_data.get("vector", ""),
        )
        enriched = {
            **nvd_data,
            "in_kev": kev_data.get("in_kev", False),
            "kev_details": kev_data,
            "techniques": mitre_data.get("techniques", []),
        }
        results.append(enriched)

    return results


# ── Consensus detection ────────────────────────────────────────────────────────

def consensus_reached(red_text: str, blue_text: str) -> bool:
    """Heuristic: if the red team concedes on all points, end early.

    We look for strong concession language without new bypass proposals.
    """
    red_lower = red_text.lower()
    blue_lower = blue_text.lower()

    blue_blocks_all = (
        blue_lower.count("blocked") + blue_lower.count("prevented") >= 3
        and "concede" not in blue_lower
    )
    red_concedes = (
        "concede" in red_lower
        and "bypass" not in red_lower
        and "alternative" not in red_lower
    )
    return blue_blocks_all and red_concedes


# ── Main debate runner ─────────────────────────────────────────────────────────

async def run_debate(cve_ids: list[str], environment: dict) -> DebateResult:
    """Execute the full Red vs Blue debate and return a structured result."""
    global_start = time.monotonic()

    # Ensure external data is loaded (no-op if already cached)
    await asyncio.gather(load_kev_catalogue(), load_attack_data())

    client = _make_openrouter_client()

    # Each agent uses the same OpenRouter client but targets its own model
    red_agent = RedTeamAgent(client)
    blue_agent = BlueTeamAgent(client)
    verdict_agent = VerdictAgent(client)

    # 1. Fetch real CVE intelligence
    logger.info("Fetching CVE data for: %s", cve_ids)
    cve_data = await fetch_all_cve_data(cve_ids)

    # 2. Red team proposes initial attack chain
    red_text, rt_in, rt_out = await red_agent.run(cve_data, environment)
    debate_history: list[dict] = [{"role": "red", "content": red_text}]
    round_latency_ms = 0

    # 3. Debate rounds
    rounds_completed = 0
    for round_num in range(MAX_ROUNDS):
        round_start = time.monotonic()

        blue_text, bt_in, bt_out = await blue_agent.run(cve_data, environment, debate_history)
        debate_history.append({"role": "blue", "content": blue_text})

        red_reply, rr_in, rr_out = await red_agent.run(cve_data, environment, debate_history)
        debate_history.append({"role": "red", "content": red_reply})

        round_latency_ms += int((time.monotonic() - round_start) * 1000)
        rounds_completed = round_num + 1

        if consensus_reached(red_reply, blue_text):
            logger.info("Early consensus after round %d", rounds_completed)
            break

    # 4. Build telemetry (verdict tokens filled in by VerdictAgent.run)
    red_tokens = red_agent.total_input_tokens + red_agent.total_output_tokens
    blue_tokens = blue_agent.total_input_tokens + blue_agent.total_output_tokens

    total_input = red_agent.total_input_tokens + blue_agent.total_input_tokens
    total_output = red_agent.total_output_tokens + blue_agent.total_output_tokens
    cost_so_far = (total_input / 1_000_000 * _INPUT_PRICE_PER_M) + (
        total_output / 1_000_000 * _OUTPUT_PRICE_PER_M
    )

    telemetry = Telemetry(
        total_cost_usd=cost_so_far,
        debate_rounds=rounds_completed,
        red_team_tokens=red_tokens,
        blue_team_tokens=blue_tokens,
        verdict_tokens=0,  # filled by verdict agent
        total_latency_ms=int((time.monotonic() - global_start) * 1000),
    )

    # 5. Verdict synthesis
    verdict = await verdict_agent.run(cve_data, environment, debate_history, telemetry)

    # Add verdict agent cost to total
    verdict_in = verdict_agent.total_input_tokens
    verdict_out = verdict_agent.total_output_tokens
    verdict_cost = (verdict_in / 1_000_000 * _INPUT_PRICE_PER_M) + (
        verdict_out / 1_000_000 * _OUTPUT_PRICE_PER_M
    )
    verdict.telemetry.total_cost_usd = cost_so_far + verdict_cost

    return DebateResult(
        cve_ids=cve_ids,
        environment=environment,
        debate_transcript=debate_history,
        verdict=verdict,
    )
