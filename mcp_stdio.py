#!/usr/bin/env python3
"""
Stdio MCP server for Glama deployment.
Speaks MCP JSON-RPC 2.0 over stdin/stdout — no uvicorn, no HTTP.
Glama runs: mcp-proxy -- python mcp_stdio.py
"""
import json
import sys
import os

# Ensure app modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import tool manifest directly from the server module
from app.mcp.server import MCP_TOOL_MANIFEST

PROTOCOL_VERSION = "2024-11-05"

SERVER_INFO = {
    "name": "meta-stamp-pockets",
    "version": "1.0.0",
    "description": (
        "Meta-Stamp Pockets \u2014 Licensed creator content for AI agents. "
        "Tiered per-pull pricing: $0.001 (generic/blogs), $0.01 (premium creator), "
        "$0.025 (flagship creators), $0.10\u2013$0.25 (verifiable institutional). "
        "85% of revenue goes directly to creators. Provenance-verified content "
        "from authorized creators."
    ),
}

TOOLS = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": tool["parameters"],
    }
    for tool in MCP_TOOL_MANIFEST["tools"]
]


def respond(result, request_id):
    msg = json.dumps({"jsonrpc": "2.0", "result": result, "id": request_id})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def error(code, message, request_id=None):
    msg = json.dumps({"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": request_id})
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def handle(data):
    method = data.get("method")
    request_id = data.get("id")

    if method == "initialize":
        respond({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }, request_id)

    elif method == "notifications/initialized":
        pass  # No response for notifications

    elif method == "tools/list":
        respond({"tools": TOOLS}, request_id)

    elif method == "tools/call":
        # Tool execution requires MongoDB — return a helpful error
        error(-32603, "Tool execution requires a live database connection. Use the HTTP endpoint at metastampv3-production.up.railway.app/mcp for live tool calls.", request_id)

    elif method == "ping":
        respond({}, request_id)

    else:
        error(-32601, f"Method not found: {method}", request_id)


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            handle(data)
        except json.JSONDecodeError:
            error(-32700, "Parse error")
        except Exception as e:
            error(-32603, str(e))


if __name__ == "__main__":
    main()
