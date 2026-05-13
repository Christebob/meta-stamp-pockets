"""KeyMap API router."""
from fastapi import APIRouter
from .routes import router as _core_router
from .ai_routes import ai_router

# Combine core and AI routes under a single router
router = APIRouter()
router.include_router(_core_router)
router.include_router(ai_router)

__all__ = ["router"]
