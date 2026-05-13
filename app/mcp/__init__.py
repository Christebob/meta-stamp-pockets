"""
META-STAMP V3 Pockets MCP (Model Context Protocol) Server.

This package implements an MCP server that allows AI agents to:
- Discover available content Pockets (list_pockets, search_pockets)
- Pull pre-indexed content in <30ms (pull_content)
- Automatically accept license terms on connection (iTunes agreement model)
- Pay creators per-pull via micro-payments

The MCP server runs as a FastAPI sub-application mounted at /mcp/.
"""
