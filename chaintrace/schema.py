from pydantic import BaseModel
from typing import List, Optional


class ConfidenceBreakdown(BaseModel):
    exploit_maturity: str
    environment_exposure: str
    defender_visibility: str
    patch_availability: str


class AttackChainStep(BaseModel):
    step_number: int
    cve_id: str = "UNKNOWN"
    action: str = ""
    blocked: bool = False
    blocking_control: Optional[str] = None


class RemediationItem(BaseModel):
    priority: str  # "critical" | "high" | "medium"
    action: str
    cve_id: str
    effort: str  # "patch" | "config_change" | "network_rule"


class Telemetry(BaseModel):
    total_cost_usd: float
    debate_rounds: int
    red_team_tokens: int
    blue_team_tokens: int
    verdict_tokens: int
    total_latency_ms: int


class VerdictOutput(BaseModel):
    attack_chain_viable: bool
    confidence_score: float
    confidence_breakdown: ConfidenceBreakdown
    attack_chain_steps: List[AttackChainStep]
    critical_gaps: List[str]
    remediation: List[RemediationItem]
    telemetry: Telemetry


class DebateResult(BaseModel):
    cve_ids: List[str]
    environment: dict
    debate_transcript: List[dict]
    verdict: VerdictOutput


class AnalyzeRequest(BaseModel):
    cve_ids: List[str]
    environment: dict
