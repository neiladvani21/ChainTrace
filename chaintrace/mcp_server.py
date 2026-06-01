"""MCP server that exposes CVE-enrichment tools to ADK agents.

Runs as a subprocess via stdio transport. Agents connect to it through
ADK's MCPToolset so they can call get_cve_details, check_cisa_kev, and
get_mitre_techniques without hallucinating CVE data.
"""

import asyncio
import json
import logging
import os
import sys

from tools.cisa import check_kev, load_kev_catalogue
from tools.mitre import get_techniques_for_cve, load_attack_data
from tools.nvd import fetch_cve

logger = logging.getLogger(__name__)

# ── MCP wire protocol helpers ──────────────────────────────────────────────────

def _ok(request_id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _err(request_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


TOOL_DEFS = [
    {
        "name": "get_cve_details",
        "description": "Fetch CVE details from NVD including severity, CVSS score, and description.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cve_id": {"type": "string", "description": "CVE identifier e.g. CVE-2021-44228"}
            },
            "required": ["cve_id"],
        },
    },
    {
        "name": "check_cisa_kev",
        "description": "Check whether a CVE appears in the CISA Known Exploited Vulnerabilities catalogue.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cve_id": {"type": "string", "description": "CVE identifier e.g. CVE-2021-44228"}
            },
            "required": ["cve_id"],
        },
    },
    {
        "name": "get_mitre_techniques",
        "description": "Map a CVE to relevant MITRE ATT&CK technique IDs based on its description and CVSS vector.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cve_id": {"type": "string", "description": "CVE identifier"},
                "description": {"type": "string", "description": "CVE description text"},
                "vector": {"type": "string", "description": "CVSS vector string"},
            },
            "required": ["cve_id"],
        },
    },
]


# ── Request handlers ───────────────────────────────────────────────────────────

async def handle_initialize(req: dict) -> dict:
    return _ok(
        req.get("id"),
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "chaintrace-mcp", "version": "1.0.0"},
        },
    )


async def handle_tools_list(req: dict) -> dict:
    return _ok(req.get("id"), {"tools": TOOL_DEFS})


async def handle_tool_call(req: dict) -> dict:
    rid = req.get("id")
    params = req.get("params", {})
    tool_name = params.get("name", "")
    args = params.get("arguments", {})

    try:
        if tool_name == "get_cve_details":
            cve_id = args.get("cve_id", "")
            api_key = os.getenv("NVD_API_KEY")
            data = await fetch_cve(cve_id, api_key)
            return _ok(rid, {"content": [{"type": "text", "text": json.dumps(data)}]})

        elif tool_name == "check_cisa_kev":
            cve_id = args.get("cve_id", "")
            data = check_kev(cve_id)
            return _ok(rid, {"content": [{"type": "text", "text": json.dumps(data)}]})

        elif tool_name == "get_mitre_techniques":
            cve_id = args.get("cve_id", "")
            description = args.get("description", "")
            vector = args.get("vector", "")
            data = get_techniques_for_cve(cve_id, description, vector)
            return _ok(rid, {"content": [{"type": "text", "text": json.dumps(data)}]})

        else:
            return _err(rid, -32601, f"Unknown tool: {tool_name}")

    except Exception as exc:
        logger.exception("Tool call error: %s", exc)
        return _err(rid, -32603, str(exc))


# ── Main stdio loop ────────────────────────────────────────────────────────────

async def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # Pre-load external data catalogues
    await asyncio.gather(load_kev_catalogue(), load_attack_data())

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_event_loop()
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    transport, _ = await loop.connect_write_pipe(asyncio.BaseProtocol, sys.stdout)

    async def send(msg: dict) -> None:
        payload = json.dumps(msg) + "\n"
        transport.write(payload.encode())

    while True:
        try:
            line = await reader.readline()
        except Exception:
            break
        if not line:
            break

        try:
            req = json.loads(line.decode().strip())
        except json.JSONDecodeError:
            continue

        method = req.get("method", "")

        if method == "initialize":
            await send(await handle_initialize(req))
        elif method == "notifications/initialized":
            pass  # no response needed
        elif method == "tools/list":
            await send(await handle_tools_list(req))
        elif method == "tools/call":
            await send(await handle_tool_call(req))
        elif "id" in req:
            await send(_err(req["id"], -32601, f"Method not found: {method}"))


if __name__ == "__main__":
    asyncio.run(main())
