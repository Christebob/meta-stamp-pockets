"""
YouTube Service Module for META-STAMP V3

Provides comprehensive YouTube Data API v3 and YouTube Analytics API integration
for the Meta-Stamp Pockets ecosystem. Supports:
- Video search and metadata retrieval
- Video uploads and management
- Channel analytics and reporting
- Playlist management
- Comment moderation

All operations require appropriate API credentials configured via environment
variables (YOUTUBE_API_KEY for public data, OAuth credentials for authenticated ops).
"""

from app.services.youtube.youtube_service import YouTubeService

__all__ = ["YouTubeService"]
