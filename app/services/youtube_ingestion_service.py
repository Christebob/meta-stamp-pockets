"""
YouTube bulk channel ingestion service for META-STAMP V3.

This service handles bulk ingestion of entire YouTube channels by:
1. Mapping the channel using YouTube Data API v3
2. Iterating through all videos with pagination
3. Extracting metadata and transcripts for each video
4. Creating a Pocket for each video with custom price_per_pull
5. Tracking ingestion job progress in MongoDB
"""

import logging
import re
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from googleapiclient.discovery import build
from pymongo import ReturnDocument
from youtube_transcript_api import YouTubeTranscriptApi

from app.config import get_settings
from app.core.database import get_db_client
from app.models.ingestion import IngestionJob, IngestionJobStatus
from app.services.pocket_service import PocketService


logger = logging.getLogger(__name__)

INGESTION_JOBS_COLLECTION = "ingestion_jobs"
YOUTUBE_PAGE_SIZE = 50


class YouTubeIngestionError(Exception):
    """Base exception for YouTube ingestion failures."""


class InvalidChannelURLError(YouTubeIngestionError):
    """Raised when a YouTube channel URL cannot be parsed."""


class ChannelNotFoundError(YouTubeIngestionError):
    """Raised when a YouTube channel cannot be found."""


class YouTubeIngestionService:
    """Service for bulk ingesting YouTube channels into Pockets."""

    def __init__(self) -> None:
        """Initialize YouTubeIngestionService with dependencies."""
        self.settings = get_settings()
        if not self.settings.youtube_api_key:
            raise YouTubeIngestionError(
                "YouTube API key not configured. Set YOUTUBE_API_KEY environment variable."
            )
        self.youtube = build("youtube", "v3", developerKey=self.settings.youtube_api_key)
        self.pocket_service = PocketService(
            compensation_per_pull=self.settings.mcp_default_price_per_pull
        )

    async def ingest_channel(
        self, channel_url: str, creator_id: str, price_per_pull: float | None = None
    ) -> IngestionJob:
        """
        Start a bulk ingestion job for a YouTube channel.

        This method creates an IngestionJob record and returns immediately.
        The actual ingestion should be run as a background task.

        Args:
            channel_url: YouTube channel URL in various formats
            creator_id: Creator's user ID
            price_per_pull: Optional custom price per pull (defaults to config value)

        Returns:
            IngestionJob with PENDING status and job_id for tracking

        Raises:
            InvalidChannelURLError: If channel URL format is invalid
            YouTubeIngestionError: If YouTube API is not configured
        """
        channel_id = self._extract_channel_id(channel_url)
        if not channel_id:
            raise InvalidChannelURLError(
                f"Could not extract channel ID from URL: {channel_url}"
            )

        # Determine price_per_pull for this job
        job_price = (
            price_per_pull
            if price_per_pull is not None
            else self.settings.mcp_default_price_per_pull
        )

        # Create initial job record
        db_client = get_db_client()
        jobs_collection = db_client.get_database()[INGESTION_JOBS_COLLECTION]

        now = datetime.now(UTC)
        job_doc: dict[str, Any] = {
            "creator_id": creator_id,
            "channel_url": channel_url,
            "channel_name": None,
            "channel_id": channel_id,
            "total_videos": 0,
            "completed": 0,
            "failed": 0,
            "status": IngestionJobStatus.PENDING.value,
            "price_per_pull": job_price,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }

        insert_result = await jobs_collection.insert_one(job_doc)
        job_doc["_id"] = str(insert_result.inserted_id)

        return self._to_job_model(job_doc)

    async def process_ingestion_job(self, job_id: str) -> IngestionJob:
        """
        Process a pending ingestion job by ingesting all videos from the channel.

        This method should be called as a background task. It:
        1. Resolves channel metadata
        2. Fetches all video IDs with pagination
        3. For each video: extracts metadata + transcript, creates Pocket
        4. Updates job progress as videos are processed

        Args:
            job_id: MongoDB ObjectId of the ingestion job

        Returns:
            IngestionJob with final status (COMPLETED or FAILED)

        Raises:
            YouTubeIngestionError: If job not found or already processed
        """
        db_client = get_db_client()
        jobs_collection = db_client.get_database()[INGESTION_JOBS_COLLECTION]

        # Fetch job record
        if not ObjectId.is_valid(job_id):
            raise YouTubeIngestionError(f"Invalid job ID format: {job_id}")

        job_doc = await jobs_collection.find_one({"_id": ObjectId(job_id)})
        if not job_doc:
            raise YouTubeIngestionError(f"Ingestion job not found: {job_id}")

        job = self._to_job_model(job_doc)

        # Ensure job is in PENDING state
        if job.status != IngestionJobStatus.PENDING:
            raise YouTubeIngestionError(
                f"Job {job_id} is not pending (status: {job.status})"
            )

        # Update status to RUNNING
        await self._update_job_status(job_id, IngestionJobStatus.RUNNING)

        try:
            # Step 1: Resolve channel metadata
            channel_info = await self._get_channel_info(job.channel_id)
            if not channel_info:
                raise ChannelNotFoundError(f"Channel not found: {job.channel_id}")

            channel_name = channel_info.get("title", "Unknown Channel")
            await self._update_job_field(job_id, "channel_name", channel_name)

            # Step 2: Fetch all video IDs from the channel
            video_ids = await self._fetch_all_video_ids(job.channel_id)
            total_videos = len(video_ids)

            if total_videos == 0:
                logger.warning(f"Channel {job.channel_id} has no videos")
                await self._update_job_field(job_id, "total_videos", 0)
                await self._update_job_status(job_id, IngestionJobStatus.COMPLETED)
                return await self.get_job_status(job_id)

            await self._update_job_field(job_id, "total_videos", total_videos)
            logger.info(
                f"Starting ingestion of {total_videos} videos from channel {channel_name}"
            )

            # Step 3: Process each video
            completed = 0
            failed = 0

            for video_id in video_ids:
                try:
                    await self._ingest_single_video(
                        video_id=video_id,
                        creator_id=job.creator_id,
                        price_per_pull=job.price_per_pull,
                    )
                    completed += 1
                except Exception as exc:
                    logger.exception(f"Failed to ingest video {video_id}: {exc}")
                    failed += 1

                # Update progress after each video
                await jobs_collection.update_one(
                    {"_id": ObjectId(job_id)},
                    {
                        "$set": {
                            "completed": completed,
                            "failed": failed,
                            "updated_at": datetime.now(UTC),
                        }
                    },
                )

            # Step 4: Mark job as completed
            await self._update_job_status(job_id, IngestionJobStatus.COMPLETED)
            logger.info(
                f"Ingestion job {job_id} completed: {completed} successful, {failed} failed"
            )

        except Exception as exc:
            logger.exception(f"Ingestion job {job_id} failed: {exc}")
            await jobs_collection.update_one(
                {"_id": ObjectId(job_id)},
                {
                    "$set": {
                        "status": IngestionJobStatus.FAILED.value,
                        "error_message": str(exc)[:2000],
                        "updated_at": datetime.now(UTC),
                    }
                },
            )

        return await self.get_job_status(job_id)

    async def get_job_status(self, job_id: str) -> IngestionJob:
        """
        Get the current status of an ingestion job.

        Args:
            job_id: MongoDB ObjectId of the ingestion job

        Returns:
            IngestionJob with current progress and status

        Raises:
            YouTubeIngestionError: If job not found
        """
        db_client = get_db_client()
        jobs_collection = db_client.get_database()[INGESTION_JOBS_COLLECTION]

        if not ObjectId.is_valid(job_id):
            raise YouTubeIngestionError(f"Invalid job ID format: {job_id}")

        job_doc = await jobs_collection.find_one({"_id": ObjectId(job_id)})
        if not job_doc:
            raise YouTubeIngestionError(f"Ingestion job not found: {job_id}")

        return self._to_job_model(job_doc)

    async def list_jobs(self, creator_id: str, limit: int = 50) -> list[IngestionJob]:
        """
        List ingestion jobs for a creator ordered by newest first.

        Args:
            creator_id: Creator's user ID
            limit: Maximum number of jobs to return (default 50)

        Returns:
            List of IngestionJob models
        """
        db_client = get_db_client()
        jobs_collection = db_client.get_database()[INGESTION_JOBS_COLLECTION]

        cursor = (
            jobs_collection.find({"creator_id": creator_id})
            .sort("created_at", -1)
            .limit(limit)
        )
        documents = await cursor.to_list(length=limit)
        return [self._to_job_model(doc) for doc in documents]

    def _extract_channel_id(self, channel_url: str) -> str | None:
        """
        Extract YouTube channel ID from various URL formats.

        Supports:
        - https://youtube.com/@DharMann
        - https://youtube.com/channel/UCxxxxxx
        - https://youtube.com/c/DharMann

        For @handle and /c/ formats, we need to resolve to channel ID via API.
        """
        channel_url = channel_url.strip()

        # Format: /channel/UCxxxxxx (direct channel ID)
        match = re.search(r"/channel/([a-zA-Z0-9_-]+)", channel_url)
        if match:
            return match.group(1)

        # Format: /@handle (need to resolve via API)
        match = re.search(r"/@([a-zA-Z0-9_-]+)", channel_url)
        if match:
            handle = match.group(1)
            return self._resolve_handle_to_channel_id(handle)

        # Format: /c/CustomName (need to resolve via API)
        match = re.search(r"/c/([a-zA-Z0-9_-]+)", channel_url)
        if match:
            custom_name = match.group(1)
            return self._resolve_custom_name_to_channel_id(custom_name)

        return None

    def _resolve_handle_to_channel_id(self, handle: str) -> str | None:
        """Resolve @handle to channel ID using YouTube Data API."""
        try:
            # Use forHandle parameter (accurate, no search quota cost)
            request = self.youtube.channels().list(
                part="snippet", forHandle=handle
            )
            response = request.execute()
            items = response.get("items", [])
            if items:
                return items[0]["id"]
            # Fallback: search API
            request = self.youtube.search().list(
                part="snippet", q=f"@{handle}", type="channel", maxResults=1
            )
            response = request.execute()
            if response.get("items"):
                return response["items"][0]["snippet"]["channelId"]
        except Exception as exc:
            logger.exception(f"Failed to resolve handle @{handle}: {exc}")
        return None

    def _resolve_custom_name_to_channel_id(self, custom_name: str) -> str | None:
        """Resolve /c/CustomName to channel ID using YouTube Data API."""
        try:
            request = self.youtube.search().list(
                part="snippet", q=custom_name, type="channel", maxResults=1
            )
            response = request.execute()
            if response.get("items"):
                return response["items"][0]["snippet"]["channelId"]
        except Exception as exc:
            logger.exception(f"Failed to resolve custom name {custom_name}: {exc}")
        return None

    async def _get_channel_info(self, channel_id: str) -> dict[str, Any] | None:
        """Fetch channel metadata from YouTube Data API."""
        try:
            request = self.youtube.channels().list(part="snippet,contentDetails", id=channel_id)
            response = request.execute()
            if response.get("items"):
                return response["items"][0]["snippet"]
        except Exception as exc:
            logger.exception(f"Failed to fetch channel info for {channel_id}: {exc}")
        return None

    async def _fetch_all_video_ids(self, channel_id: str) -> list[str]:
        """
        Fetch all video IDs from a channel using pagination.

        Uses the channel's uploads playlist and iterates through pages.
        """
        video_ids: list[str] = []

        try:
            # Get the uploads playlist ID
            request = self.youtube.channels().list(part="contentDetails", id=channel_id)
            response = request.execute()
            if not response.get("items"):
                return video_ids

            uploads_playlist_id = response["items"][0]["contentDetails"]["relatedPlaylists"][
                "uploads"
            ]

            # Paginate through all videos
            next_page_token = None
            while True:
                request = self.youtube.playlistItems().list(
                    part="contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=YOUTUBE_PAGE_SIZE,
                    pageToken=next_page_token,
                )
                response = request.execute()

                for item in response.get("items", []):
                    video_id = item["contentDetails"]["videoId"]
                    video_ids.append(video_id)

                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

        except Exception as exc:
            logger.exception(f"Failed to fetch video IDs for channel {channel_id}: {exc}")

        return video_ids

    async def _ingest_single_video(
        self, video_id: str, creator_id: str, price_per_pull: float
    ) -> None:
        """
        Ingest a single YouTube video by creating a Pocket with metadata + transcript.

        Args:
            video_id: YouTube video ID
            creator_id: Creator's user ID
            price_per_pull: Price per pull for this Pocket
        """
        # Fetch video metadata
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        metadata = await self._get_video_metadata(video_id)

        # Fetch transcript
        transcript_text = await self._get_video_transcript(video_id)

        # Build snapshot text
        snapshot_parts = []
        if metadata:
            if metadata.get("title"):
                snapshot_parts.append(f"Title: {metadata['title']}")
            if metadata.get("description"):
                snapshot_parts.append(f"Description: {metadata['description']}")
            if metadata.get("publishedAt"):
                snapshot_parts.append(f"Published: {metadata['publishedAt']}")

        if transcript_text:
            snapshot_parts.append(f"Transcript:\n{transcript_text}")

        snapshot_text = "\n\n".join(snapshot_parts)

        # Create Pocket directly in MongoDB (bypassing URLProcessorService)
        # This is more efficient for bulk operations
        db_client = get_db_client()
        pockets_collection = db_client.get_pockets_collection()

        now = datetime.now(UTC)
        pocket_doc: dict[str, Any] = {
            "creator_id": creator_id,
            "content_url": video_url,
            "content_type": "youtube",
            "status": "active",
            "pull_count": 0,
            "compensation_earned": 0.0,
            "snapshot_text": snapshot_text if snapshot_text else None,
            "error_message": None,
            "source_metadata": {
                "platform": "youtube",
                "video_id": video_id,
                "price_per_pull": price_per_pull,
                **(metadata or {}),
            },
            "created_at": now,
            "updated_at": now,
        }

        await pockets_collection.insert_one(pocket_doc)
        logger.debug(f"Created Pocket for video {video_id}")

    async def _get_video_metadata(self, video_id: str) -> dict[str, Any] | None:
        """Fetch video metadata from YouTube Data API."""
        try:
            request = self.youtube.videos().list(part="snippet,contentDetails", id=video_id)
            response = request.execute()
            if response.get("items"):
                item = response["items"][0]
                snippet = item["snippet"]
                content_details = item["contentDetails"]
                return {
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "publishedAt": snippet.get("publishedAt"),
                    "channelTitle": snippet.get("channelTitle"),
                    "duration": content_details.get("duration"),
                    "thumbnails": snippet.get("thumbnails", {}).get("default", {}).get("url"),
                }
        except Exception as exc:
            logger.warning(f"Failed to fetch metadata for video {video_id}: {exc}")
        return None

    async def _get_video_transcript(self, video_id: str) -> str | None:
        """
        Fetch video transcript using youtube-transcript-api.

        Returns concatenated transcript text or None if unavailable.
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([entry["text"] for entry in transcript_list])
            return transcript_text
        except Exception as exc:
            logger.warning(f"Failed to fetch transcript for video {video_id}: {exc}")
        return None

    async def _update_job_status(self, job_id: str, status: IngestionJobStatus) -> None:
        """Update job status in MongoDB."""
        db_client = get_db_client()
        jobs_collection = db_client.get_database()[INGESTION_JOBS_COLLECTION]
        await jobs_collection.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {"status": status.value, "updated_at": datetime.now(UTC)}},
        )

    async def _update_job_field(self, job_id: str, field: str, value: Any) -> None:
        """Update a specific field in the job document."""
        db_client = get_db_client()
        jobs_collection = db_client.get_database()[INGESTION_JOBS_COLLECTION]
        await jobs_collection.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {field: value, "updated_at": datetime.now(UTC)}},
        )

    def _to_job_model(self, document: dict[str, Any]) -> IngestionJob:
        """Convert MongoDB document to IngestionJob model."""
        serialized = dict(document)
        if "_id" in serialized:
            serialized["_id"] = str(serialized["_id"])
        return IngestionJob.model_validate(serialized)


__all__ = [
    "YouTubeIngestionService",
    "YouTubeIngestionError",
    "InvalidChannelURLError",
    "ChannelNotFoundError",
]
