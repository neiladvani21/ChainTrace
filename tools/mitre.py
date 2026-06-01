"""MITRE ATT&CK technique mapper.

Downloads the enterprise ATT&CK STIX bundle once at startup, then maps
CVSS attack-vector strings and keywords to the most relevant technique IDs.
"""

import logging
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/"
    "enterprise-attack/enterprise-attack.json"
)

# Cached technique list: [{id, name, description, url}]
_techniques: list[dict] = []


async def load_attack_data() -> None:
    """Download and cache the MITRE ATT&CK enterprise bundle. Call once at startup."""
    global _techniques
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(ATTACK_URL)
            resp.raise_for_status()
            bundle = resp.json()

        for obj in bundle.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            if obj.get("revoked") or obj.get("x_mitre_deprecated"):
                continue

            ext_refs = obj.get("external_references", [])
            tid = ""
            url = ""
            for ref in ext_refs:
                if ref.get("source_name") == "mitre-attack":
                    tid = ref.get("external_id", "")
                    url = ref.get("url", "")
                    break

            if not tid:
                continue

            _techniques.append(
                {
                    "id": tid,
                    "name": obj.get("name", ""),
                    "description": (obj.get("description") or "")[:300],
                    "url": url,
                    "platforms": obj.get("x_mitre_platforms", []),
                    "detection": (obj.get("x_mitre_detection") or "")[:200],
                }
            )

        logger.info("Loaded %d ATT&CK techniques", len(_techniques))
    except Exception as exc:
        logger.warning("Failed to load MITRE ATT&CK data: %s", exc)


# Heuristic keyword → technique ID mappings for common CVE patterns
_VECTOR_HINTS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"AV:N", re.I), "T1190", "Exploit Public-Facing Application"),
    (re.compile(r"JNDI|log4j|log4shell", re.I), "T1190", "Exploit Public-Facing Application"),
    (re.compile(r"deseri[az]", re.I), "T1059", "Command and Scripting Interpreter"),
    (re.compile(r"spring|rce|remote.code", re.I), "T1190", "Exploit Public-Facing Application"),
    (re.compile(r"AV:L|local.priv", re.I), "T1068", "Exploitation for Privilege Escalation"),
    (re.compile(r"sql.inject", re.I), "T1190", "Exploit Public-Facing Application"),
    (re.compile(r"path.trav|directory.trav", re.I), "T1083", "File and Directory Discovery"),
    (re.compile(r"ldap|dns.exfil|exfiltrat", re.I), "T1048", "Exfiltration Over Alternative Protocol"),
    (re.compile(r"command.inject|OS.command", re.I), "T1059", "Command and Scripting Interpreter"),
    (re.compile(r"upload|webshell|web.shell", re.I), "T1505", "Server Software Component"),
]


def get_techniques_for_cve(cve_id: str, description: str, vector: str) -> dict:
    """Map a CVE to relevant ATT&CK techniques based on description and CVSS vector."""
    combined_text = f"{cve_id} {description} {vector}"
    matched: list[dict] = []
    seen_ids: set[str] = set()

    for pattern, tid, default_name in _VECTOR_HINTS:
        if pattern.search(combined_text) and tid not in seen_ids:
            seen_ids.add(tid)
            # Try to find the full technique record from the loaded data
            tech = next((t for t in _techniques if t["id"] == tid), None)
            if tech:
                matched.append(tech)
            else:
                matched.append({"id": tid, "name": default_name, "description": "", "url": ""})

    # Always include T1190 for network-reachable CVEs as a baseline
    if "T1190" not in seen_ids and ("AV:N" in vector or not vector):
        tech = next((t for t in _techniques if t["id"] == "T1190"), None)
        if tech:
            matched.append(tech)

    return {
        "cve_id": cve_id,
        "techniques": matched[:5],  # cap at 5 most relevant
    }
