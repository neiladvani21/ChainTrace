"""NVD CVE data fetcher using nvdlib. No API key required for low-volume demo use."""

import asyncio
import logging
from typing import Optional

import nvdlib

logger = logging.getLogger(__name__)


async def fetch_cve(cve_id: str, api_key: Optional[str] = None) -> dict:
    """Fetch CVE details from NVD via nvdlib.

    nvdlib enforces a 6-second sleep between requests when no API key is
    provided. We run the blocking call in a thread so the event loop stays
    free during that wait.
    """
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: nvdlib.searchCVE(cveId=cve_id, key=api_key) if api_key else nvdlib.searchCVE(cveId=cve_id),
        )
    except Exception as exc:
        logger.warning("NVD fetch failed for %s: %s", cve_id, exc)
        return _fallback(cve_id)

    if not results:
        logger.warning("No NVD results for %s", cve_id)
        return _fallback(cve_id)

    cve = results[0]
    return _parse(cve)


def _parse(cve) -> dict:
    """Extract the fields we care about from an nvdlib CVE object."""
    description = ""
    try:
        description = cve.descriptions[0].value
    except (AttributeError, IndexError):
        pass

    severity = getattr(cve, "v31severity", None) or getattr(cve, "v30severity", None) or "UNKNOWN"
    score = getattr(cve, "v31score", None) or getattr(cve, "v30score", None) or 0.0
    vector = getattr(cve, "v31vector", None) or getattr(cve, "v30vector", None) or ""
    published = str(getattr(cve, "published", ""))

    return {
        "id": cve.id,
        "description": description,
        "severity": severity,
        "score": score,
        "vector": vector,
        "published": published,
    }


def _fallback(cve_id: str) -> dict:
    """Return a minimal stub when NVD is unreachable."""
    return {
        "id": cve_id,
        "description": "Could not retrieve CVE data from NVD.",
        "severity": "UNKNOWN",
        "score": 0.0,
        "vector": "",
        "published": "",
    }
