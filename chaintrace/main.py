"""ChainTrace FastAPI application entry point.

Exposes:
  POST /analyze  — run the full CVE debate and return a DebateResult
  GET  /health   — liveness check
  GET  /         — serve the frontend HTML
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from orchestrator import run_debate
from schema import AnalyzeRequest, DebateResult
from tools.cisa import load_kev_catalogue
from tools.mitre import load_attack_data

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ChainTrace",
    description="CVE attack-chain debate system — Red Team vs Blue Team",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"


@app.on_event("startup")
async def startup() -> None:
    """Pre-fetch external threat intelligence catalogues at startup."""
    logger.info("Loading CISA KEV and MITRE ATT&CK catalogues...")
    await load_kev_catalogue()
    await load_attack_data()
    logger.info("Catalogues loaded — ChainTrace ready")


@app.get("/")
async def serve_frontend() -> FileResponse:
    """Serve the single-page frontend."""
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(FRONTEND_PATH, media_type="text/html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=DebateResult)
async def analyze(request: AnalyzeRequest) -> DebateResult:
    """Run the Red vs Blue debate across the supplied CVEs and environment.

    Validates inputs, delegates to the orchestrator, and returns the full
    DebateResult including transcript and structured verdict.
    """
    if not request.cve_ids:
        raise HTTPException(status_code=422, detail="At least one CVE ID is required")
    if len(request.cve_ids) > 5:
        raise HTTPException(status_code=422, detail="Maximum 5 CVE IDs per request")

    # Normalize IDs to uppercase
    cve_ids = [c.strip().upper() for c in request.cve_ids if c.strip()]

    logger.info("Starting debate for CVEs: %s", cve_ids)
    try:
        result = await run_debate(cve_ids, request.environment)
    except RuntimeError as exc:
        # Surfaces missing OPENROUTER_API_KEY clearly to the caller
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during debate")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
