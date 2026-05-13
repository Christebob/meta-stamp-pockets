"""
YouTube API Routes for META-STAMP V3

REST endpoints for YouTube Data API v3 and YouTube Analytics API operations.
Provides both public data access and authenticated operations via OAuth 2.0.

Endpoints:
    Public (API key):
        GET  /api/v1/youtube/health          - Health check
        GET  /api/v1/youtube/search          - Search videos
        GET  /api/v1/youtube/videos/{id}     - Get video details
        GET  /api/v1/youtube/channels/{id}   - Get channel info

    OAuth:
        GET  /api/v1/youtube/auth/url        - Get OAuth authorization URL
        POST /api/v1/youtube/auth/callback   - Exchange auth code for tokens

    Authenticated (requires OAuth tokens):
        POST /api/v1/youtube/upload          - Upload a video
        GET  /api/v1/youtube/my/channel      - Get authenticated user's channel
        GET  /api/v1/youtube/my/videos       - Get authenticated user's videos
        PUT  /api/v1/youtube/videos/{id}     - Update video metadata
        DELETE /api/v1/youtube/videos/{id}   - Delete a video
        GET  /api/v1/youtube/analytics/channel  - Channel analytics
        GET  /api/v1/youtube/analytics/video/{id} - Video analytics
        GET  /api/v1/youtube/analytics/top   - Top performing videos
        POST /api/v1/youtube/playlists       - Create playlist
        POST /api/v1/youtube/playlists/{id}/videos - Add video to playlist
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Header
from pydantic import BaseModel

from app.config import get_settings
from app.services.youtube.youtube_service import YouTubeService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/youtube", tags=["YouTube"])


# =============================================================================
# Request/Response Models
# =============================================================================


class OAuthCallbackRequest(BaseModel):
    """Request model for OAuth callback."""
    code: str


class VideoUpdateRequest(BaseModel):
    """Request model for updating video metadata."""
    title: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    privacy_status: str | None = None


class PlaylistCreateRequest(BaseModel):
    """Request model for creating a playlist."""
    title: str
    description: str = ""
    privacy_status: str = "private"


class PlaylistAddVideoRequest(BaseModel):
    """Request model for adding a video to a playlist."""
    video_id: str


# =============================================================================
# Helper Functions
# =============================================================================


def _get_youtube_service() -> YouTubeService:
    """Create and return a configured YouTubeService instance."""
    settings = get_settings()
    return YouTubeService(
        api_key=settings.youtube_api_key,
        client_id=settings.google_oauth_client_id,
        client_secret=settings.google_oauth_client_secret,
        redirect_uri=settings.google_oauth_redirect_uri,
    )


def _extract_token_data(authorization: str | None) -> dict[str, Any]:
    """Extract OAuth token data from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header with Bearer token required for this operation",
        )
    # In production, decode the token and retrieve stored credentials
    # For now, we expect the token to be the access_token
    return {"access_token": authorization.replace("Bearer ", "")}


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health")
async def youtube_health():
    """Check YouTube API configuration status."""
    settings = get_settings()
    return {
        "status": "ok",
        "api_key_configured": settings.has_youtube_configured,
        "oauth_configured": settings.has_youtube_oauth_configured,
    }


# =============================================================================
# Public Data Endpoints (API Key)
# =============================================================================


@router.get("/search")
async def search_videos(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=50, description="Max results"),
    order: str = Query("relevance", description="Sort order"),
):
    """Search for YouTube videos."""
    service = _get_youtube_service()
    try:
        results = service.search_videos(query=q, max_results=max_results, order=order)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos/{video_id}")
async def get_video_details(video_id: str):
    """Get detailed information about a specific video."""
    service = _get_youtube_service()
    try:
        result = service.get_video_details(video_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube video details error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/channels/{channel_id}")
async def get_channel_info(channel_id: str):
    """Get information about a YouTube channel."""
    service = _get_youtube_service()
    try:
        result = service.get_channel_info(channel_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube channel info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# OAuth Endpoints
# =============================================================================


@router.get("/auth/url")
async def get_oauth_url():
    """Get the OAuth 2.0 authorization URL for YouTube access."""
    service = _get_youtube_service()
    settings = get_settings()

    if not settings.has_youtube_oauth_configured:
        raise HTTPException(
            status_code=503,
            detail="YouTube OAuth credentials not configured",
        )

    try:
        url = service.get_oauth_url()
        return {"authorization_url": url}
    except Exception as e:
        logger.error(f"OAuth URL generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/callback")
async def oauth_callback(request: OAuthCallbackRequest):
    """Exchange authorization code for OAuth tokens."""
    service = _get_youtube_service()
    try:
        tokens = service.exchange_code(request.code)
        return {"success": True, "tokens": tokens}
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Authenticated Endpoints (OAuth Required)
# =============================================================================


@router.get("/my/channel")
async def get_my_channel(authorization: str = Header(None)):
    """Get the authenticated user's YouTube channel."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        result = service.get_my_channel(token_data)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"My channel error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my/videos")
async def get_my_videos(
    max_results: int = Query(25, ge=1, le=50),
    authorization: str = Header(None),
):
    """Get the authenticated user's uploaded videos."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        results = service.get_my_videos(token_data, max_results=max_results)
        return {"videos": results, "count": len(results)}
    except Exception as e:
        logger.error(f"My videos error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/videos/{video_id}")
async def update_video(
    video_id: str,
    request: VideoUpdateRequest,
    authorization: str = Header(None),
):
    """Update a video's metadata."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        result = service.update_video(
            token_data,
            video_id=video_id,
            title=request.title,
            description=request.description,
            tags=request.tags,
            privacy_status=request.privacy_status,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/videos/{video_id}")
async def delete_video(
    video_id: str,
    authorization: str = Header(None),
):
    """Delete a video from YouTube."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        result = service.delete_video(token_data, video_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete video error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Analytics Endpoints (OAuth Required)
# =============================================================================


@router.get("/analytics/channel")
async def get_channel_analytics(
    start_date: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str | None = Query(None, description="End date YYYY-MM-DD"),
    authorization: str = Header(None),
):
    """Get channel-level analytics data."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        result = service.get_channel_analytics(
            token_data, start_date=start_date, end_date=end_date
        )
        return result
    except Exception as e:
        logger.error(f"Channel analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/video/{video_id}")
async def get_video_analytics(
    video_id: str,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    authorization: str = Header(None),
):
    """Get analytics data for a specific video."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        result = service.get_video_analytics(
            token_data, video_id, start_date=start_date, end_date=end_date
        )
        return result
    except Exception as e:
        logger.error(f"Video analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/top")
async def get_top_videos(
    max_results: int = Query(10, ge=1, le=50),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    authorization: str = Header(None),
):
    """Get top performing videos by views."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        results = service.get_top_videos(
            token_data, max_results=max_results,
            start_date=start_date, end_date=end_date,
        )
        return {"videos": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Top videos error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Playlist Endpoints (OAuth Required)
# =============================================================================


@router.post("/playlists")
async def create_playlist(
    request: PlaylistCreateRequest,
    authorization: str = Header(None),
):
    """Create a new YouTube playlist."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        result = service.create_playlist(
            token_data,
            title=request.title,
            description=request.description,
            privacy_status=request.privacy_status,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create playlist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/playlists/{playlist_id}/videos")
async def add_video_to_playlist(
    playlist_id: str,
    request: PlaylistAddVideoRequest,
    authorization: str = Header(None),
):
    """Add a video to a playlist."""
    token_data = _extract_token_data(authorization)
    service = _get_youtube_service()
    try:
        result = service.add_video_to_playlist(
            token_data,
            playlist_id=playlist_id,
            video_id=request.video_id,
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add to playlist error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
