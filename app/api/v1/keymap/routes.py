"""
KeyMap API Routes — Meta-Stamp V3

Endpoints:
  GET  /keymap/apps                          → List all supported apps
  GET  /keymap/apps/{app_id}/shortcuts       → Get all shortcuts for an app
  GET  /keymap/apps/{app_id}/categories      → Get shortcut categories for an app
  POST /keymap/map                           → Map shortcuts from source to target app
  POST /keymap/export                        → Export mapping as a downloadable file
  GET  /keymap/health                        → Health check
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional

from app.services.keymap import (
    get_all_apps,
    get_shortcuts_for_app,
    get_categories_for_app,
    map_shortcuts,
    export_mapping,
    APP_METADATA,
)

router = APIRouter(prefix="/keymap", tags=["KeyMap"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MapRequest(BaseModel):
    source_app: str = Field(..., description="Source app ID (e.g. 'pro_tools')")
    target_app: str = Field(..., description="Target app ID (e.g. 'final_cut_pro')")
    platform: Literal["mac", "windows"] = Field("mac", description="Operating system platform")
    category_filter: Optional[str] = Field(None, description="Filter by category (optional)")


class ExportRequest(BaseModel):
    source_app: str = Field(..., description="Source app ID")
    target_app: str = Field(..., description="Target app ID")
    platform: Literal["mac", "windows"] = Field("mac", description="Operating system platform")
    format: Literal["autohotkey", "karabiner", "keyboard_maestro", "stream_deck", "markdown"] = Field(
        ..., description="Export format"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def keymap_health():
    """Health check for the KeyMap module."""
    return {
        "status": "healthy",
        "module": "KeyMap",
        "supported_apps": len(APP_METADATA),
        "export_formats": ["autohotkey", "karabiner", "keyboard_maestro", "stream_deck", "markdown"],
    }


@router.get("/apps")
async def list_apps():
    """
    List all supported creative applications.

    Returns metadata for every app in the KeyMap database including
    app type (DAW/NLE), vendor, supported platforms, and description.
    """
    return {
        "count": len(APP_METADATA),
        "apps": get_all_apps(),
    }


@router.get("/apps/{app_id}/shortcuts")
async def get_shortcuts(
    app_id: str,
    platform: Literal["mac", "windows"] = Query("mac", description="Filter by platform"),
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """
    Get all keyboard shortcuts for a specific application.

    Returns the full shortcut list, optionally filtered by platform and category.
    """
    if app_id not in APP_METADATA:
        raise HTTPException(
            status_code=404,
            detail=f"App '{app_id}' not found. Use GET /keymap/apps to see supported apps.",
        )

    shortcuts = get_shortcuts_for_app(app_id)

    # Filter by platform — exclude shortcuts with no mapping for this platform
    shortcuts = [s for s in shortcuts if s.get(platform) is not None]

    # Filter by category
    if category:
        shortcuts = [s for s in shortcuts if s["category"].lower() == category.lower()]

    return {
        "app_id": app_id,
        "app_name": APP_METADATA[app_id]["name"],
        "platform": platform,
        "category_filter": category,
        "count": len(shortcuts),
        "shortcuts": shortcuts,
    }


@router.get("/apps/{app_id}/categories")
async def get_categories(app_id: str):
    """Get all shortcut categories for a specific application."""
    if app_id not in APP_METADATA:
        raise HTTPException(
            status_code=404,
            detail=f"App '{app_id}' not found.",
        )
    return {
        "app_id": app_id,
        "categories": get_categories_for_app(app_id),
    }


@router.post("/map")
async def map_app_shortcuts(request: MapRequest):
    """
    Map keyboard shortcuts from one application to another.

    Takes a source app and target app, and returns a complete mapping
    of every action — showing the source shortcut, the equivalent target
    shortcut, and whether a mapping exists.

    This is the core engine of the KeyMap system. Use the results to:
    - Understand which shortcuts transfer directly
    - Identify gaps where no equivalent exists
    - Feed into the /export endpoint to generate remapping scripts
    """
    if request.source_app not in APP_METADATA:
        raise HTTPException(status_code=404, detail=f"Source app '{request.source_app}' not found.")
    if request.target_app not in APP_METADATA:
        raise HTTPException(status_code=404, detail=f"Target app '{request.target_app}' not found.")

    mappings = map_shortcuts(request.source_app, request.target_app, request.platform)

    # Apply category filter
    if request.category_filter:
        mappings = [m for m in mappings if m["category"].lower() == request.category_filter.lower()]

    mapped_count = sum(1 for m in mappings if m["mapped"])
    total_count = len(mappings)

    return {
        "source_app": request.source_app,
        "source_name": APP_METADATA[request.source_app]["name"],
        "target_app": request.target_app,
        "target_name": APP_METADATA[request.target_app]["name"],
        "platform": request.platform,
        "total_shortcuts": total_count,
        "mapped_shortcuts": mapped_count,
        "unmapped_shortcuts": total_count - mapped_count,
        "coverage_pct": round(mapped_count / total_count * 100, 1) if total_count > 0 else 0,
        "mappings": mappings,
    }


@router.post("/export")
async def export_shortcuts(request: ExportRequest):
    """
    Export a shortcut mapping as a downloadable remapping script.

    Supported formats:
    - **autohotkey**: Windows AutoHotkey v2 script (.ahk) — app-specific, runs in background
    - **karabiner**: macOS Karabiner-Elements JSON (.json) — import into Karabiner complex rules
    - **keyboard_maestro**: macOS Keyboard Maestro macro library (.kmmacros) — double-click to import
    - **stream_deck**: Elgato Stream Deck profile JSON (.streamDeckProfile) — import into Stream Deck app
    - **markdown**: Human-readable reference table (.md) — print or share
    """
    if request.source_app not in APP_METADATA:
        raise HTTPException(status_code=404, detail=f"Source app '{request.source_app}' not found.")
    if request.target_app not in APP_METADATA:
        raise HTTPException(status_code=404, detail=f"Target app '{request.target_app}' not found.")

    mappings = map_shortcuts(request.source_app, request.target_app, request.platform)
    source_name = APP_METADATA[request.source_app]["name"]
    target_name = APP_METADATA[request.target_app]["name"]

    try:
        content, filename, mime_type = export_mapping(
            mappings=mappings,
            source_app=request.source_app,
            target_app=request.target_app,
            source_name=source_name,
            target_name=target_name,
            format=request.format,
            platform=request.platform,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return Response(
        content=content.encode("utf-8"),
        media_type=mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-KeyMap-Source": request.source_app,
            "X-KeyMap-Target": request.target_app,
            "X-KeyMap-Format": request.format,
        },
    )


@router.get("/export/formats")
async def list_export_formats():
    """List all available export formats with descriptions."""
    return {
        "formats": [
            {
                "id": "autohotkey",
                "name": "AutoHotkey v2",
                "platform": "windows",
                "extension": ".ahk",
                "description": "Windows background script — remaps keys only when target app is focused.",
                "install_url": "https://www.autohotkey.com/",
            },
            {
                "id": "karabiner",
                "name": "Karabiner-Elements",
                "platform": "mac",
                "extension": ".json",
                "description": "macOS low-level key remapper — import as complex_modifications rule.",
                "install_url": "https://karabiner-elements.pqrs.org/",
            },
            {
                "id": "keyboard_maestro",
                "name": "Keyboard Maestro",
                "platform": "mac",
                "extension": ".kmmacros",
                "description": "macOS macro engine — double-click to import macro group.",
                "install_url": "https://www.keyboardmaestro.com/",
            },
            {
                "id": "stream_deck",
                "name": "Elgato Stream Deck",
                "platform": "mac+windows",
                "extension": ".streamDeckProfile",
                "description": "Stream Deck hardware profile — import into Stream Deck app.",
                "install_url": "https://www.elgato.com/stream-deck",
            },
            {
                "id": "markdown",
                "name": "Markdown Reference",
                "platform": "mac+windows",
                "extension": ".md",
                "description": "Human-readable shortcut reference table — print or share.",
                "install_url": None,
            },
        ]
    }
