"""
KeyMap AI Routes — Codex-powered shortcut intelligence.

Endpoints:
  POST /keymap/ai/explain        → Explain what a shortcut does
  POST /keymap/ai/fill-gaps      → AI suggestions for unmapped shortcuts
  POST /keymap/ai/custom-app     → Generate shortcuts for any unlisted app
  POST /keymap/ai/transition     → Workflow transition guide
  POST /keymap/ai/stream-deck    → Optimal Stream Deck layout suggestion
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal

from app.services.keymap import map_shortcuts, APP_METADATA
from app.services.keymap.codex_prompts import (
    build_gap_fill_prompt,
    build_explain_shortcut_prompt,
    build_custom_app_prompt,
    build_workflow_transition_prompt,
    build_stream_deck_layout_prompt,
    call_codex,
)

ai_router = APIRouter(prefix="/keymap/ai", tags=["KeyMap AI"])


class ExplainRequest(BaseModel):
    app_name: str = Field(..., description="Name of the application")
    shortcut: str = Field(..., description="Shortcut to explain, e.g. 'Cmd+E'")
    action_label: str = Field(..., description="Action label, e.g. 'Split Clip'")
    context: Optional[str] = Field(None, description="Optional workflow context")


class FillGapsRequest(BaseModel):
    source_app: str = Field(..., description="Source app ID")
    target_app: str = Field(..., description="Target app ID")
    platform: Literal["mac", "windows"] = Field("mac")


class CustomAppRequest(BaseModel):
    app_name: str = Field(..., description="Name of the app to generate shortcuts for")
    platform: Literal["mac", "windows"] = Field("mac")
    categories: Optional[list[str]] = Field(None, description="Categories to include")


class TransitionRequest(BaseModel):
    source_app: str = Field(..., description="Source app name or ID")
    target_app: str = Field(..., description="Target app name or ID")
    user_role: str = Field("creative professional", description="User's role (e.g. 'audio engineer')")
    platform: Literal["mac", "windows"] = Field("mac")


class StreamDeckRequest(BaseModel):
    target_app: str = Field(..., description="App to build the layout for")
    deck_size: str = Field("15-key", description="Stream Deck model size")
    user_role: str = Field("video editor", description="User's role")


@ai_router.post("/explain")
async def explain_shortcut(request: ExplainRequest):
    """Use AI to explain what a keyboard shortcut does in plain language."""
    prompt = build_explain_shortcut_prompt(
        app_name=request.app_name,
        shortcut=request.shortcut,
        action_label=request.action_label,
        context=request.context,
    )
    response = await call_codex(prompt, temperature=0.4)
    return {"explanation": response}


@ai_router.post("/fill-gaps")
async def fill_mapping_gaps(request: FillGapsRequest):
    """
    Use AI to suggest equivalents for shortcuts that have no direct mapping.
    Identifies gaps in the KeyMap database and asks Codex to fill them.
    """
    if request.source_app not in APP_METADATA:
        raise HTTPException(status_code=404, detail=f"Source app '{request.source_app}' not found.")
    if request.target_app not in APP_METADATA:
        raise HTTPException(status_code=404, detail=f"Target app '{request.target_app}' not found.")

    mappings = map_shortcuts(request.source_app, request.target_app, request.platform)
    unmapped = [m for m in mappings if not m["mapped"]]

    if not unmapped:
        return {
            "message": "No gaps found — all shortcuts have direct mappings!",
            "suggestions": [],
        }

    source_name = APP_METADATA[request.source_app]["name"]
    target_name = APP_METADATA[request.target_app]["name"]

    prompt = build_gap_fill_prompt(source_name, target_name, unmapped, request.platform)
    response = await call_codex(prompt, temperature=0.2, max_tokens=3000)

    return {
        "source_app": request.source_app,
        "target_app": request.target_app,
        "gap_count": len(unmapped),
        "ai_suggestions_raw": response,
    }


@ai_router.post("/custom-app")
async def generate_custom_app_shortcuts(request: CustomAppRequest):
    """
    Generate a shortcut database for any creative app not in our built-in database.
    Powered by Codex — returns a JSON shortcut list that can be used immediately.
    """
    prompt = build_custom_app_prompt(
        app_name=request.app_name,
        platform=request.platform,
        categories=request.categories,
    )
    response = await call_codex(prompt, temperature=0.2, max_tokens=4000)
    return {
        "app_name": request.app_name,
        "platform": request.platform,
        "shortcuts_raw": response,
        "note": "Parse the JSON array to use these shortcuts in the KeyMap system.",
    }


@ai_router.post("/transition")
async def workflow_transition_guide(request: TransitionRequest):
    """
    Generate a practical workflow transition guide for switching between two apps.
    Includes muscle-memory differences, paradigm shifts, and a 2-week practice plan.
    """
    # Resolve app names from IDs if provided
    source_name = APP_METADATA.get(request.source_app, {}).get("name", request.source_app)
    target_name = APP_METADATA.get(request.target_app, {}).get("name", request.target_app)

    prompt = build_workflow_transition_prompt(
        source_app=source_name,
        target_app=target_name,
        user_role=request.user_role,
        platform=request.platform,
    )
    response = await call_codex(prompt, temperature=0.5, max_tokens=3000)
    return {
        "source_app": source_name,
        "target_app": target_name,
        "user_role": request.user_role,
        "guide": response,
    }


@ai_router.post("/stream-deck")
async def stream_deck_layout(request: StreamDeckRequest):
    """
    Generate an optimal Stream Deck layout for a specific app and user role.
    Returns a 15-key layout with action assignments and rationale.
    """
    app_name = APP_METADATA.get(request.target_app, {}).get("name", request.target_app)
    prompt = build_stream_deck_layout_prompt(
        target_app=app_name,
        deck_size=request.deck_size,
        user_role=request.user_role,
    )
    response = await call_codex(prompt, temperature=0.4, max_tokens=2000)
    return {
        "target_app": app_name,
        "deck_size": request.deck_size,
        "user_role": request.user_role,
        "layout_raw": response,
    }
