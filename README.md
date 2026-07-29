# Meta-Stamp Pockets

**Attribution and settlement infrastructure for AI agent access.**

AI agents consume creator content at machine speed. Meta-Stamp Pockets is the payment and licensing rail at the point of access: every request either settles payment to the creator or generates a timestamped denial record — a documented licensing opportunity, declined. License, or leave a paper trail.

## What It Does

- Serves licensed, provenance-tracked creator content to AI agents via the Model Context Protocol (MCP) and HTTP 402
- Per-pull pricing set by the creator — $1.00 default, configurable per asset
- 85% of every pull goes to the creator, automatically
- Every access attempt lands in an immutable event ledger — paid pulls settle, refusals leave evidence

## How Content Gets Protected

Two registration paths, each matched to how content arrives:

- **Direct upload** — the file is perceptually fingerprinted (audio spectral analysis, image/video hashing, text signatures) and then discarded. The fingerprint travels; the file stays with the creator.
- **YouTube URL** — ownership is verified through OAuth channel connection; the pocket carries metadata, transcript, and the access ledger. YouTube remains the canonical host.

Three protection layers are filed IP (12 U.S. provisional patent filings, sole inventor). Two run live today: multi-modal fingerprinting and the immutable access ledger. Embedded watermarking is the roadmap layer.

## MCP Server

Connect AI agents to Meta-Stamp Pockets via the Model Context Protocol:

```
https://metastampv3-production.up.railway.app/mcp
```

**Available tools:**

- `search_pockets` — search licensed creator content by keyword
- `pull_content` — retrieve provenance-verified content from a Pocket
- `list_pockets` — list all available Pockets, optionally filtered by creator

**Authentication:** Bearer token required. Register at https://metastampv3-production.up.railway.app/docs

## Live Demo

https://metastampv3-production.up.railway.app/demo

## API Docs

https://metastampv3-production.up.railway.app/docs

## For Creators

If you're a creator interested in licensing your content through Meta-Stamp Pockets, contact chriscoynetalent@gmail.com.

## For Enterprise

If you represent an AI platform interested in licensing access to the Meta-Stamp creator network, contact chriscoynetalent@gmail.com.

---

*Meta-Stamp, LLC · Simi Valley, CA · Patent pending*
