"""CISA Known Exploited Vulnerabilities (KEV) checker.

The full KEV catalogue is fetched once at startup and cached in memory.
Individual CVE lookups are then O(1) set membership checks.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

KEV_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

# Module-level cache populated by load_kev_catalogue()
_kev_set: set[str] = set()
_kev_details: dict[str, dict] = {}


async def load_kev_catalogue() -> None:
    """Download and cache the CISA KEV catalogue. Call once at startup."""
    global _kev_set, _kev_details
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(KEV_URL)
            resp.raise_for_status()
            data = resp.json()

        for entry in data.get("vulnerabilities", []):
            cve_id = entry.get("cveID", "")
            if cve_id:
                _kev_set.add(cve_id)
                _kev_details[cve_id] = {
                    "vendor_project": entry.get("vendorProject", ""),
                    "product": entry.get("product", ""),
                    "vulnerability_name": entry.get("vulnerabilityName", ""),
                    "date_added": entry.get("dateAdded", ""),
                    "required_action": entry.get("requiredAction", ""),
                    "due_date": entry.get("dueDate", ""),
                }
        logger.info("Loaded %d entries from CISA KEV catalogue", len(_kev_set))
    except Exception as exc:
        logger.warning("Failed to load CISA KEV catalogue: %s", exc)


def check_kev(cve_id: str) -> dict:
    """Return KEV status and details for a single CVE ID."""
    if cve_id in _kev_set:
        return {
            "cve_id": cve_id,
            "in_kev": True,
            **_kev_details.get(cve_id, {}),
        }
    return {
        "cve_id": cve_id,
        "in_kev": False,
    }
