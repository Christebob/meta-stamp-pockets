"""
Discovery API endpoints for META-STAMP V3.

Endpoints:
- GET /discovery/lookup: Look up a pocket by content URL for AI agent discovery
"""

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.core.database import get_db_client


logger = logging.getLogger(__name__)
router = APIRouter(tags=["discovery"])

MCP_ENDPOINT = "https://metastampv3-production.up.railway.app/mcp"


@router.get(
    "/lookup",
    summary="Look up a licensed pocket by content URL",
    description=(
        "Given a content URL, returns the corresponding Pocket ID, creator wallet "
        "address, and price per pull. AI agents use this endpoint to discover "
        "licensing terms before accessing creator content via MCP."
    ),
)
async def discovery_lookup(
    url: str = Query(..., description="The content URL to look up"),
) -> dict:
    """
    Look up a pocket by content URL.

    Used by AI agents to discover licensing information for a given URL
    before making a pull request via the MCP endpoint.

    Args:
        url: The content URL to look up

    Returns:
        dict: pocket_id, creator_wallet_address, and price_per_pull

    Raises:
        HTTPException: 404 if no pocket found for the given URL
    """
    if not url or not url.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL parameter is required",
        )

    normalized_url = url.strip()

    try:
        db_client = get_db_client()
        db = db_client.get_database()

        # Look up pocket by content_url
        pocket = await db["pockets"].find_one(
            {"content_url": normalized_url, "status": "active"}
        )

        if not pocket:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active pocket found for URL: {normalized_url}",
            )

        # Look up creator wallet
        creator_id = str(pocket.get("creator_id", ""))
        wallet_address = ""

        if creator_id:
            wallet = await db["wallets"].find_one({"creator_id": creator_id})
            if wallet:
                wallet_address = str(wallet.get("_id", ""))

        return {
            "pocket_id": str(pocket["_id"]),
            "creator_wallet_address": wallet_address,
            "price_per_pull": pocket.get("price_per_pull", 0.0025),
            "content_url": normalized_url,
            "mcp_endpoint": MCP_ENDPOINT,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Discovery lookup failed for URL %s: %s", normalized_url, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Discovery lookup failed",
        ) from exc
