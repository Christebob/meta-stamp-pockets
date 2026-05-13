"""
YouTube Service for META-STAMP V3

Comprehensive YouTube Data API v3 and YouTube Analytics API integration.
Handles both public data access (via API key) and authenticated operations
(via OAuth 2.0) including video uploads, channel management, and analytics.

Dependencies:
    - google-api-python-client: YouTube Data API client
    - google-auth-oauthlib: OAuth 2.0 authentication flow
    - google-auth-httplib2: HTTP transport for Google APIs

Author: META-STAMP V3 Development Team
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

# OAuth 2.0 scopes for YouTube operations
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly",
]

# Valid video privacy statuses
PRIVACY_STATUSES = ("public", "private", "unlisted")

# Valid video categories (subset of YouTube category IDs)
VIDEO_CATEGORIES = {
    "1": "Film & Animation",
    "2": "Autos & Vehicles",
    "10": "Music",
    "15": "Pets & Animals",
    "17": "Sports",
    "19": "Travel & Events",
    "20": "Gaming",
    "22": "People & Blogs",
    "23": "Comedy",
    "24": "Entertainment",
    "25": "News & Politics",
    "26": "Howto & Style",
    "27": "Education",
    "28": "Science & Technology",
    "29": "Nonprofits & Activism",
}


class YouTubeService:
    """
    Service class for YouTube Data API v3 and YouTube Analytics API operations.

    Provides two modes of operation:
    1. Public data access (API key only): search, video info, channel info
    2. Authenticated access (OAuth 2.0): uploads, analytics, channel management

    Example usage:
        ```python
        from app.services.youtube import YouTubeService

        # Public data access
        service = YouTubeService(api_key="your-google-api-key")
        results = service.search_videos("meta stamp creator tools")

        # Authenticated access
        service = YouTubeService(
            api_key="your-google-api-key",
            client_id="417807...",
            client_secret="your-google-client-secret",
            redirect_uri="http://localhost:3000/auth/google/callback"
        )
        ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        redirect_uri: str = "http://localhost:3000/auth/google/callback",
    ):
        self.api_key = api_key
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._youtube = None
        self._youtube_analytics = None
        self._credentials = None

    # =========================================================================
    # Service Initialization
    # =========================================================================

    def _get_public_client(self):
        """Get YouTube Data API client for public data access (API key only)."""
        if self._youtube is None:
            if not self.api_key:
                raise ValueError("YouTube API key is required for public data access")
            self._youtube = build("youtube", "v3", developerKey=self.api_key)
        return self._youtube

    def _get_authenticated_client(self, credentials: Credentials):
        """Get YouTube Data API client for authenticated operations."""
        return build("youtube", "v3", credentials=credentials)

    def _get_analytics_client(self, credentials: Credentials):
        """Get YouTube Analytics API client for reporting."""
        return build("youtubeAnalytics", "v2", credentials=credentials)

    def get_oauth_url(self) -> str:
        """
        Generate the OAuth 2.0 authorization URL for user consent.

        Returns:
            Authorization URL to redirect the user to for Google sign-in.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("OAuth client ID and secret are required")

        flow = InstalledAppFlow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri],
                }
            },
            scopes=YOUTUBE_SCOPES,
        )
        flow.redirect_uri = self.redirect_uri
        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            prompt="consent",
        )
        return auth_url

    def exchange_code(self, authorization_code: str) -> dict[str, Any]:
        """
        Exchange an authorization code for OAuth credentials.

        Args:
            authorization_code: The code returned from the OAuth consent flow.

        Returns:
            Dictionary containing access_token, refresh_token, and expiry.
        """
        flow = InstalledAppFlow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri],
                }
            },
            scopes=YOUTUBE_SCOPES,
        )
        flow.redirect_uri = self.redirect_uri
        flow.fetch_token(code=authorization_code)
        credentials = flow.credentials

        return {
            "access_token": credentials.token,
            "refresh_token": credentials.refresh_token,
            "token_uri": credentials.token_uri,
            "client_id": credentials.client_id,
            "client_secret": credentials.client_secret,
            "expiry": credentials.expiry.isoformat() if credentials.expiry else None,
        }

    def _credentials_from_tokens(self, token_data: dict[str, Any]) -> Credentials:
        """Build Credentials object from stored token data."""
        return Credentials(
            token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.client_id,
            client_secret=self.client_secret,
        )

    # =========================================================================
    # Public Data Operations (API Key)
    # =========================================================================

    def search_videos(
        self,
        query: str,
        max_results: int = 10,
        order: str = "relevance",
        video_type: str = "any",
    ) -> list[dict[str, Any]]:
        """
        Search for YouTube videos.

        Args:
            query: Search query string.
            max_results: Maximum number of results (1-50).
            order: Sort order (relevance, date, rating, viewCount, title).
            video_type: Filter by type (any, episode, movie).

        Returns:
            List of video result dictionaries with id, title, description,
            thumbnail, channel info, and publish date.
        """
        youtube = self._get_public_client()
        try:
            request = youtube.search().list(
                part="snippet",
                q=query,
                maxResults=min(max_results, 50),
                order=order,
                type="video",
                videoType=video_type,
            )
            response = request.execute()

            results = []
            for item in response.get("items", []):
                results.append({
                    "video_id": item["id"]["videoId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url"),
                    "channel_id": item["snippet"]["channelId"],
                    "channel_title": item["snippet"]["channelTitle"],
                    "published_at": item["snippet"]["publishedAt"],
                })
            return results

        except HttpError as e:
            logger.error(f"YouTube search error: {e}")
            raise

    def get_video_details(self, video_id: str) -> dict[str, Any]:
        """
        Get detailed information about a specific video.

        Args:
            video_id: YouTube video ID.

        Returns:
            Dictionary with full video details including statistics,
            content details, and status.
        """
        youtube = self._get_public_client()
        try:
            request = youtube.videos().list(
                part="snippet,contentDetails,statistics,status",
                id=video_id,
            )
            response = request.execute()

            if not response.get("items"):
                return {"error": f"Video {video_id} not found"}

            item = response["items"][0]
            return {
                "video_id": video_id,
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "channel_id": item["snippet"]["channelId"],
                "channel_title": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "tags": item["snippet"].get("tags", []),
                "category_id": item["snippet"].get("categoryId"),
                "duration": item["contentDetails"]["duration"],
                "definition": item["contentDetails"]["definition"],
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "like_count": int(item["statistics"].get("likeCount", 0)),
                "comment_count": int(item["statistics"].get("commentCount", 0)),
                "privacy_status": item["status"]["privacyStatus"],
                "thumbnails": item["snippet"]["thumbnails"],
            }

        except HttpError as e:
            logger.error(f"YouTube video details error: {e}")
            raise

    def get_channel_info(self, channel_id: str) -> dict[str, Any]:
        """
        Get information about a YouTube channel.

        Args:
            channel_id: YouTube channel ID.

        Returns:
            Dictionary with channel details and statistics.
        """
        youtube = self._get_public_client()
        try:
            request = youtube.channels().list(
                part="snippet,contentDetails,statistics,brandingSettings",
                id=channel_id,
            )
            response = request.execute()

            if not response.get("items"):
                return {"error": f"Channel {channel_id} not found"}

            item = response["items"][0]
            return {
                "channel_id": channel_id,
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "custom_url": item["snippet"].get("customUrl"),
                "published_at": item["snippet"]["publishedAt"],
                "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url"),
                "subscriber_count": int(item["statistics"].get("subscriberCount", 0)),
                "video_count": int(item["statistics"].get("videoCount", 0)),
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
            }

        except HttpError as e:
            logger.error(f"YouTube channel info error: {e}")
            raise

    # =========================================================================
    # Authenticated Operations (OAuth 2.0)
    # =========================================================================

    def upload_video(
        self,
        token_data: dict[str, Any],
        file_path: str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        category_id: str = "22",
        privacy_status: str = "private",
    ) -> dict[str, Any]:
        """
        Upload a video to YouTube.

        Args:
            token_data: OAuth token data from exchange_code().
            file_path: Local path to the video file.
            title: Video title (max 100 characters).
            description: Video description (max 5000 characters).
            tags: List of video tags.
            category_id: YouTube category ID (default: "22" People & Blogs).
            privacy_status: Privacy status (public, private, unlisted).

        Returns:
            Dictionary with upload result including video ID and URL.
        """
        if privacy_status not in PRIVACY_STATUSES:
            raise ValueError(f"Invalid privacy status. Must be one of: {PRIVACY_STATUSES}")

        credentials = self._credentials_from_tokens(token_data)
        youtube = self._get_authenticated_client(credentials)

        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags or [],
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            file_path,
            chunksize=256 * 1024,
            resumable=True,
        )

        try:
            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"Upload progress: {int(status.progress() * 100)}%")

            video_id = response["id"]
            logger.info(f"Video uploaded successfully: {video_id}")

            return {
                "success": True,
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": title,
                "privacy_status": privacy_status,
            }

        except HttpError as e:
            logger.error(f"YouTube upload error: {e}")
            return {"success": False, "error": str(e)}

    def get_my_channel(self, token_data: dict[str, Any]) -> dict[str, Any]:
        """
        Get the authenticated user's channel information.

        Args:
            token_data: OAuth token data from exchange_code().

        Returns:
            Dictionary with the user's channel details and statistics.
        """
        credentials = self._credentials_from_tokens(token_data)
        youtube = self._get_authenticated_client(credentials)

        try:
            request = youtube.channels().list(
                part="snippet,contentDetails,statistics",
                mine=True,
            )
            response = request.execute()

            if not response.get("items"):
                return {"error": "No channel found for authenticated user"}

            item = response["items"][0]
            return {
                "channel_id": item["id"],
                "title": item["snippet"]["title"],
                "description": item["snippet"]["description"],
                "subscriber_count": int(item["statistics"].get("subscriberCount", 0)),
                "video_count": int(item["statistics"].get("videoCount", 0)),
                "view_count": int(item["statistics"].get("viewCount", 0)),
                "uploads_playlist_id": item["contentDetails"]["relatedPlaylists"]["uploads"],
            }

        except HttpError as e:
            logger.error(f"YouTube my channel error: {e}")
            raise

    def get_my_videos(
        self,
        token_data: dict[str, Any],
        max_results: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Get the authenticated user's uploaded videos.

        Args:
            token_data: OAuth token data from exchange_code().
            max_results: Maximum number of videos to return.

        Returns:
            List of video dictionaries with details and statistics.
        """
        channel = self.get_my_channel(token_data)
        if "error" in channel:
            return []

        playlist_id = channel["uploads_playlist_id"]
        credentials = self._credentials_from_tokens(token_data)
        youtube = self._get_authenticated_client(credentials)

        try:
            request = youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=min(max_results, 50),
            )
            response = request.execute()

            videos = []
            for item in response.get("items", []):
                videos.append({
                    "video_id": item["contentDetails"]["videoId"],
                    "title": item["snippet"]["title"],
                    "description": item["snippet"]["description"],
                    "thumbnail": item["snippet"]["thumbnails"].get("high", {}).get("url"),
                    "published_at": item["snippet"]["publishedAt"],
                    "position": item["snippet"]["position"],
                })
            return videos

        except HttpError as e:
            logger.error(f"YouTube my videos error: {e}")
            raise

    def update_video(
        self,
        token_data: dict[str, Any],
        video_id: str,
        title: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        privacy_status: str | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing video's metadata.

        Args:
            token_data: OAuth token data from exchange_code().
            video_id: YouTube video ID to update.
            title: New title (optional).
            description: New description (optional).
            tags: New tags (optional).
            privacy_status: New privacy status (optional).

        Returns:
            Dictionary with updated video details.
        """
        credentials = self._credentials_from_tokens(token_data)
        youtube = self._get_authenticated_client(credentials)

        try:
            # First get current video details
            current = youtube.videos().list(
                part="snippet,status",
                id=video_id,
            ).execute()

            if not current.get("items"):
                return {"error": f"Video {video_id} not found"}

            item = current["items"][0]
            snippet = item["snippet"]
            status = item["status"]

            # Update only provided fields
            if title is not None:
                snippet["title"] = title[:100]
            if description is not None:
                snippet["description"] = description[:5000]
            if tags is not None:
                snippet["tags"] = tags

            body = {"id": video_id, "snippet": snippet}

            if privacy_status is not None:
                if privacy_status not in PRIVACY_STATUSES:
                    raise ValueError(f"Invalid privacy status: {privacy_status}")
                status["privacyStatus"] = privacy_status
                body["status"] = status

            parts = "snippet"
            if "status" in body:
                parts += ",status"

            response = youtube.videos().update(
                part=parts,
                body=body,
            ).execute()

            return {
                "success": True,
                "video_id": response["id"],
                "title": response["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={response['id']}",
            }

        except HttpError as e:
            logger.error(f"YouTube update video error: {e}")
            return {"success": False, "error": str(e)}

    def delete_video(
        self,
        token_data: dict[str, Any],
        video_id: str,
    ) -> dict[str, Any]:
        """
        Delete a video from YouTube.

        Args:
            token_data: OAuth token data from exchange_code().
            video_id: YouTube video ID to delete.

        Returns:
            Dictionary with deletion result.
        """
        credentials = self._credentials_from_tokens(token_data)
        youtube = self._get_authenticated_client(credentials)

        try:
            youtube.videos().delete(id=video_id).execute()
            logger.info(f"Video deleted: {video_id}")
            return {"success": True, "video_id": video_id}

        except HttpError as e:
            logger.error(f"YouTube delete video error: {e}")
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Analytics Operations (OAuth 2.0)
    # =========================================================================

    def get_channel_analytics(
        self,
        token_data: dict[str, Any],
        start_date: str | None = None,
        end_date: str | None = None,
        metrics: str = "views,estimatedMinutesWatched,averageViewDuration,likes,subscribersGained",
    ) -> dict[str, Any]:
        """
        Get channel-level analytics data.

        Args:
            token_data: OAuth token data from exchange_code().
            start_date: Start date in YYYY-MM-DD format (default: 28 days ago).
            end_date: End date in YYYY-MM-DD format (default: today).
            metrics: Comma-separated list of metrics to retrieve.

        Returns:
            Dictionary with analytics data including column headers and rows.
        """
        credentials = self._credentials_from_tokens(token_data)
        analytics = self._get_analytics_client(credentials)

        if not start_date:
            start_date = (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            request = analytics.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics=metrics,
                dimensions="day",
                sort="day",
            )
            response = request.execute()

            return {
                "start_date": start_date,
                "end_date": end_date,
                "metrics": metrics.split(","),
                "columns": [h["name"] for h in response.get("columnHeaders", [])],
                "rows": response.get("rows", []),
                "totals": response.get("totals", {}),
            }

        except HttpError as e:
            logger.error(f"YouTube analytics error: {e}")
            raise

    def get_video_analytics(
        self,
        token_data: dict[str, Any],
        video_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Get analytics data for a specific video.

        Args:
            token_data: OAuth token data from exchange_code().
            video_id: YouTube video ID.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            Dictionary with video-level analytics data.
        """
        credentials = self._credentials_from_tokens(token_data)
        analytics = self._get_analytics_client(credentials)

        if not start_date:
            start_date = (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            request = analytics.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,averageViewDuration,likes,comments",
                filters=f"video=={video_id}",
                dimensions="day",
                sort="day",
            )
            response = request.execute()

            return {
                "video_id": video_id,
                "start_date": start_date,
                "end_date": end_date,
                "columns": [h["name"] for h in response.get("columnHeaders", [])],
                "rows": response.get("rows", []),
            }

        except HttpError as e:
            logger.error(f"YouTube video analytics error: {e}")
            raise

    def get_top_videos(
        self,
        token_data: dict[str, Any],
        max_results: int = 10,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get top performing videos by views.

        Args:
            token_data: OAuth token data from exchange_code().
            max_results: Number of top videos to return.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            List of dictionaries with video analytics sorted by views.
        """
        credentials = self._credentials_from_tokens(token_data)
        analytics = self._get_analytics_client(credentials)

        if not start_date:
            start_date = (datetime.now() - timedelta(days=28)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        try:
            request = analytics.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views,estimatedMinutesWatched,likes,subscribersGained",
                dimensions="video",
                sort="-views",
                maxResults=max_results,
            )
            response = request.execute()

            videos = []
            for row in response.get("rows", []):
                videos.append({
                    "video_id": row[0],
                    "views": row[1],
                    "watch_time_minutes": row[2],
                    "likes": row[3],
                    "subscribers_gained": row[4],
                })
            return videos

        except HttpError as e:
            logger.error(f"YouTube top videos error: {e}")
            raise

    # =========================================================================
    # Playlist Operations (OAuth 2.0)
    # =========================================================================

    def create_playlist(
        self,
        token_data: dict[str, Any],
        title: str,
        description: str = "",
        privacy_status: str = "private",
    ) -> dict[str, Any]:
        """
        Create a new YouTube playlist.

        Args:
            token_data: OAuth token data from exchange_code().
            title: Playlist title.
            description: Playlist description.
            privacy_status: Privacy status (public, private, unlisted).

        Returns:
            Dictionary with created playlist details.
        """
        credentials = self._credentials_from_tokens(token_data)
        youtube = self._get_authenticated_client(credentials)

        try:
            request = youtube.playlists().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": title,
                        "description": description,
                    },
                    "status": {
                        "privacyStatus": privacy_status,
                    },
                },
            )
            response = request.execute()

            return {
                "success": True,
                "playlist_id": response["id"],
                "title": response["snippet"]["title"],
                "url": f"https://www.youtube.com/playlist?list={response['id']}",
            }

        except HttpError as e:
            logger.error(f"YouTube create playlist error: {e}")
            return {"success": False, "error": str(e)}

    def add_video_to_playlist(
        self,
        token_data: dict[str, Any],
        playlist_id: str,
        video_id: str,
    ) -> dict[str, Any]:
        """
        Add a video to a playlist.

        Args:
            token_data: OAuth token data from exchange_code().
            playlist_id: YouTube playlist ID.
            video_id: YouTube video ID to add.

        Returns:
            Dictionary with result of the operation.
        """
        credentials = self._credentials_from_tokens(token_data)
        youtube = self._get_authenticated_client(credentials)

        try:
            request = youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id,
                        },
                    },
                },
            )
            response = request.execute()

            return {
                "success": True,
                "playlist_item_id": response["id"],
                "video_id": video_id,
                "playlist_id": playlist_id,
            }

        except HttpError as e:
            logger.error(f"YouTube add to playlist error: {e}")
            return {"success": False, "error": str(e)}
