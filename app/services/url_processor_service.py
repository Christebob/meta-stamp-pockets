"""
URL Processor Service Module for META-STAMP V3

This service handles extraction of content from various URL sources including:
- YouTube videos: Transcript extraction using youtube-transcript-api and metadata scraping
- Vimeo videos: Metadata extraction via page scraping
- General webpages: Text content extraction via BeautifulSoup

Security Constraints (NON-NEGOTIABLE - Agent Action Plan Section 0.3):
- URLs must pass validation (http/https scheme only)
- URLs pointing to dangerous file types (.zip, .exe, etc.) are REJECTED
- Platform detection for proper routing

Author: META-STAMP V3 Development Team
"""

import contextlib
import json
import logging
import re

from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from bs4 import BeautifulSoup
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from app.utils.file_validator import validate_url as validator_validate_url


# =============================================================================
# CONSTANTS
# =============================================================================

# Default timeout for HTTP requests (in seconds)
DEFAULT_REQUEST_TIMEOUT: int = 30

# YouTube video ID standard length
YOUTUBE_VIDEO_ID_LENGTH: int = 11

# User-Agent header for HTTP requests to avoid being blocked
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# YouTube URL patterns for video ID extraction
YOUTUBE_WATCH_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)
YOUTUBE_SHORT_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)
YOUTUBE_EMBED_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)
YOUTUBE_SHORTS_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)

# Vimeo URL patterns for video ID extraction
VIMEO_STANDARD_PATTERN = re.compile(
    r"^(?:https?://)?(?:www\.)?vimeo\.com/(\d+)",
    re.IGNORECASE,
)
VIMEO_PLAYER_PATTERN = re.compile(
    r"^(?:https?://)?player\.vimeo\.com/video/(\d+)",
    re.IGNORECASE,
)


# =============================================================================
# URL PROCESSOR SERVICE CLASS
# =============================================================================


class URLProcessorService:
    """
    Service for extracting content from URLs including YouTube, Vimeo, and web pages.

    This service provides comprehensive URL processing capabilities:
    - Platform detection (YouTube, Vimeo, or general webpage)
    - YouTube transcript extraction and metadata retrieval
    - Vimeo metadata extraction
    - General webpage text content extraction

    Security:
    - All URLs are validated before processing
    - Dangerous file type URLs are rejected
    - HTTP timeouts prevent hanging requests

    Example:
        >>> service = URLProcessorService()
        >>> result = await service.process_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        >>> print(result["platform"])
        "youtube"
    """

    def __init__(self, request_timeout: int = DEFAULT_REQUEST_TIMEOUT) -> None:
        """
        Initialize the URL Processor Service.

        Args:
            request_timeout: Timeout in seconds for HTTP requests (default: 30)
        """
        self.logger = logging.getLogger(__name__)
        self.request_timeout = request_timeout
        self._session_headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    # =========================================================================
    # URL VALIDATION
    # =========================================================================

    def validate_url(self, url: str) -> bool:
        """
        Validate URL for content import security.

        Uses the centralized URL validation from file_validator module to ensure
        consistent security enforcement across all URL uploads per Agent Action
        Plan section 0.3 NON-NEGOTIABLE security constraints.

        Args:
            url: The URL to validate

        Returns:
            True if URL is valid and safe for processing, False otherwise

        Security:
            - Only http and https schemes are allowed
            - URLs pointing to dangerous file types (.zip, .exe, etc.) are rejected
            - Malformed URLs are rejected
        """
        is_valid, url_type, error_message = validator_validate_url(url)

        if not is_valid:
            self.logger.warning(f"URL validation failed: {error_message}")
            return False

        self.logger.debug(f"URL validated successfully: platform={url_type}")
        return True

    # =========================================================================
    # PLATFORM DETECTION
    # =========================================================================

    def detect_platform(self, url: str) -> str:
        """
        Detect the platform type from a URL.

        Identifies whether a URL points to YouTube, Vimeo, or a general webpage
        for routing to the appropriate processing method.

        Args:
            url: The URL to analyze

        Returns:
            Platform identifier: "youtube", "vimeo", or "webpage"

        Example:
            >>> service = URLProcessorService()
            >>> service.detect_platform("https://www.youtube.com/watch?v=abc123")
            "youtube"
            >>> service.detect_platform("https://vimeo.com/123456789")
            "vimeo"
            >>> service.detect_platform("https://example.com/article")
            "webpage"
        """
        if not url:
            return "webpage"

        url_lower = url.lower().strip()

        # Check for YouTube URLs
        parsed = urlparse(url_lower)
        netloc = parsed.netloc.replace("www.", "").replace("m.", "")

        if netloc in ["youtube.com", "youtu.be"]:
            return "youtube"

        # Check for Vimeo URLs
        if "vimeo.com" in netloc:
            return "vimeo"

        # Default to general webpage
        return "webpage"

    # =========================================================================
    # VIDEO ID EXTRACTION
    # =========================================================================

    def extract_youtube_video_id(self, url: str) -> str | None:
        """
        Extract the video ID from a YouTube URL.

        Supports multiple YouTube URL formats:
        - Standard watch URLs: youtube.com/watch?v=VIDEO_ID
        - Short URLs: youtu.be/VIDEO_ID
        - Embed URLs: youtube.com/embed/VIDEO_ID
        - Shorts URLs: youtube.com/shorts/VIDEO_ID

        Args:
            url: The YouTube URL to parse

        Returns:
            11-character video ID string, or None if extraction fails

        Example:
            >>> service = URLProcessorService()
            >>> service.extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            "dQw4w9WgXcQ"
            >>> service.extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ")
            "dQw4w9WgXcQ"
        """
        if not url:
            return None

        url = url.strip()

        # Try standard watch URL pattern
        match = YOUTUBE_WATCH_PATTERN.search(url)
        if match:
            return match.group(1)

        # Try short URL pattern (youtu.be)
        match = YOUTUBE_SHORT_PATTERN.search(url)
        if match:
            return match.group(1)

        # Try embed URL pattern
        match = YOUTUBE_EMBED_PATTERN.search(url)
        if match:
            return match.group(1)

        # Try shorts URL pattern
        match = YOUTUBE_SHORTS_PATTERN.search(url)
        if match:
            return match.group(1)

        # Fallback: Try to extract 'v' parameter from query string
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            video_id_list = query_params.get("v", [])
            if video_id_list and len(video_id_list[0]) == YOUTUBE_VIDEO_ID_LENGTH:
                return video_id_list[0]
        except Exception as e:
            self.logger.debug(f"Query string parsing failed: {e}")

        self.logger.warning(f"Could not extract YouTube video ID from URL: {url}")
        return None

    def extract_vimeo_video_id(self, url: str) -> str | None:
        """
        Extract the video ID from a Vimeo URL.

        Supports multiple Vimeo URL formats:
        - Standard URLs: vimeo.com/VIDEO_ID
        - Player embed URLs: player.vimeo.com/video/VIDEO_ID

        Args:
            url: The Vimeo URL to parse

        Returns:
            Numeric video ID string, or None if extraction fails

        Example:
            >>> service = URLProcessorService()
            >>> service.extract_vimeo_video_id("https://vimeo.com/123456789")
            "123456789"
        """
        if not url:
            return None

        url = url.strip()

        # Try standard Vimeo URL pattern
        match = VIMEO_STANDARD_PATTERN.search(url)
        if match:
            return match.group(1)

        # Try player embed URL pattern
        match = VIMEO_PLAYER_PATTERN.search(url)
        if match:
            return match.group(1)

        # Fallback: Try to extract numeric ID from path
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            for part in path_parts:
                if part.isdigit():
                    return part
        except Exception as e:
            self.logger.debug(f"Path parsing failed: {e}")

        self.logger.warning(f"Could not extract Vimeo video ID from URL: {url}")
        return None

    # =========================================================================
    # YOUTUBE PROCESSING
    # =========================================================================

    async def process_youtube_url(self, url: str) -> dict[str, Any]:
        """
        Process a YouTube URL to extract transcript and metadata.

        This method:
        1. Extracts the video ID from the URL
        2. Fetches the transcript using youtube-transcript-api
        3. Combines transcript segments into full text
        4. Scrapes video page for metadata (title, description, duration, views)

        Args:
            url: The YouTube URL to process

        Returns:
            Dictionary containing:
            - success: Boolean indicating successful processing
            - platform: "youtube"
            - url: Original URL
            - video_id: Extracted video ID
            - transcript: Full text transcript (if available)
            - transcript_segments: Raw transcript data with timestamps
            - metadata: Video metadata (title, description, duration, views)
            - error: Error message if processing failed

        Example:
            >>> service = URLProcessorService()
            >>> result = await service.process_youtube_url(
            ...     "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            ... )
            >>> print(result["transcript"][:50])
            "We're no strangers to love..."
        """
        result: dict[str, Any] = {
            "success": False,
            "platform": "youtube",
            "url": url,
            "video_id": None,
            "transcript": None,
            "transcript_segments": None,
            "metadata": {},
            "error": None,
            "processed_at": datetime.now(tz=UTC).isoformat(),
        }

        # Extract video ID
        video_id = self.extract_youtube_video_id(url)
        if not video_id:
            result["error"] = "Could not extract video ID from YouTube URL"
            self.logger.error(result["error"])
            return result

        result["video_id"] = video_id
        self.logger.info(f"Processing YouTube video: {video_id}")

        # Attempt to fetch transcript
        transcript_text = None
        transcript_segments = None

        try:
            # Try to get transcript in preferred languages
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get manually created transcript first (higher quality)
            transcript = None
            try:
                # Try English first
                transcript = transcript_list.find_manually_created_transcript(["en"])
            except NoTranscriptFound:
                try:
                    # Fall back to auto-generated
                    transcript = transcript_list.find_generated_transcript(["en"])
                except NoTranscriptFound:
                    try:
                        # Try any available transcript
                        for t in transcript_list:
                            transcript = t
                            break
                    except Exception:
                        pass

            if transcript:
                transcript_data = transcript.fetch()
                transcript_segments = transcript_data

                # Combine transcript segments into full text
                transcript_text = " ".join(segment.get("text", "") for segment in transcript_data)
                # Clean up transcript text
                transcript_text = self._clean_transcript_text(transcript_text)

                result["transcript"] = transcript_text
                result["transcript_segments"] = transcript_segments
                self.logger.info(
                    f"Successfully extracted transcript for video {video_id} "
                    f"({len(transcript_text)} characters)"
                )

        except TranscriptsDisabled:
            result["error"] = "Transcripts are disabled for this video"
            self.logger.warning(f"Transcripts disabled for video: {video_id}")

        except VideoUnavailable:
            result["error"] = "Video is unavailable"
            self.logger.warning(f"Video unavailable: {video_id}")

        except NoTranscriptFound:
            result["error"] = "No transcript found for this video"
            self.logger.warning(f"No transcript found for video: {video_id}")

        except Exception as e:
            self.logger.exception("Error fetching YouTube transcript")
            result["error"] = f"Error fetching transcript: {e!s}"

        # Attempt to fetch video metadata via page scraping
        try:
            metadata = await self._fetch_youtube_metadata(url, video_id)
            result["metadata"] = metadata
        except Exception as e:
            self.logger.warning(f"Error fetching YouTube metadata: {e}")
            result["metadata"] = {"title": None, "description": None}

        # Mark as successful if we got either transcript or metadata
        result["success"] = bool(transcript_text or result["metadata"].get("title"))

        return result

    async def _fetch_youtube_metadata(self, url: str, video_id: str) -> dict[str, Any]:
        """
        Fetch YouTube video metadata by scraping the video page.

        Args:
            url: The YouTube URL
            video_id: The extracted video ID

        Returns:
            Dictionary containing title, description, channel, and other metadata
        """
        metadata: dict[str, Any] = {
            "title": None,
            "description": None,
            "channel": None,
            "duration": None,
            "view_count": None,
            "upload_date": None,
        }

        try:
            response = requests.get(
                url,
                headers=self._session_headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title and description using helper methods
            metadata["title"] = self._extract_youtube_title(soup)
            metadata["description"] = self._extract_youtube_description(soup)

            # Extract channel name from JSON-LD if available
            script_tags = soup.find_all("script", type="application/ld+json")
            for script in script_tags:
                try:
                    if script.string is None:
                        continue
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get("@type") == "VideoObject":
                        metadata["channel"] = data.get("author", {}).get("name")
                        metadata["duration"] = data.get("duration")
                        metadata["upload_date"] = data.get("uploadDate")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue

            # Try to extract view count from page content
            view_count_match = re.search(r'"viewCount":\s*"(\d+)"', response.text)
            if view_count_match:
                with contextlib.suppress(ValueError):
                    metadata["view_count"] = int(view_count_match.group(1))

            self.logger.debug(f"Extracted YouTube metadata: title={metadata.get('title')}")

        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout fetching YouTube metadata for video: {video_id}")

        except requests.exceptions.ConnectionError:
            self.logger.warning(f"Connection error fetching YouTube metadata for video: {video_id}")

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Request error fetching YouTube metadata: {e}")

        except Exception:
            self.logger.exception("Unexpected error fetching YouTube metadata")

        return metadata

    def _extract_youtube_title(self, soup: BeautifulSoup) -> str | None:
        """
        Extract YouTube video title from page soup.

        Args:
            soup: BeautifulSoup object of the page

        Returns:
            Extracted title string or None
        """
        # Try og:title meta tag first
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            content = og_title["content"]
            return str(content) if content else None

        # Fallback to <title> tag
        title_tag = soup.find("title")
        if title_tag:
            title_text = title_tag.get_text()
            # Remove " - YouTube" suffix
            if " - YouTube" in title_text:
                title_text = title_text.rsplit(" - YouTube", 1)[0]
            return title_text.strip()

        return None

    def _extract_youtube_description(self, soup: BeautifulSoup) -> str | None:
        """
        Extract YouTube video description from page soup.

        Args:
            soup: BeautifulSoup object of the page

        Returns:
            Extracted description string or None
        """
        # Try og:description meta tag first
        og_description = soup.find("meta", property="og:description")
        if og_description and og_description.get("content"):
            content = og_description["content"]
            return str(content) if content else None

        # Fallback to meta description
        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description and meta_description.get("content"):
            content = meta_description["content"]
            return str(content) if content else None

        return None

    def _clean_transcript_text(self, text: str) -> str:
        """
        Clean transcript text by removing extra whitespace and formatting.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned transcript text
        """
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove multiple spaces
        return re.sub(r" {2,}", " ", text)

    # =========================================================================
    # VIMEO PROCESSING
    # =========================================================================

    async def process_vimeo_url(self, url: str) -> dict[str, Any]:
        """
        Process a Vimeo URL to extract video metadata.

        Note: Vimeo does not provide public transcript access, so this method
        focuses on extracting available metadata from the video page.

        Args:
            url: The Vimeo URL to process

        Returns:
            Dictionary containing:
            - success: Boolean indicating successful processing
            - platform: "vimeo"
            - url: Original URL
            - video_id: Extracted video ID
            - metadata: Video metadata (title, description, duration, author)
            - error: Error message if processing failed

        Example:
            >>> service = URLProcessorService()
            >>> result = await service.process_vimeo_url("https://vimeo.com/123456789")
            >>> print(result["metadata"]["title"])
            "My Awesome Video"
        """
        result: dict[str, Any] = {
            "success": False,
            "platform": "vimeo",
            "url": url,
            "video_id": None,
            "metadata": {},
            "error": None,
            "processed_at": datetime.now(tz=UTC).isoformat(),
        }

        # Extract video ID
        video_id = self.extract_vimeo_video_id(url)
        if not video_id:
            result["error"] = "Could not extract video ID from Vimeo URL"
            self.logger.error(result["error"])
            return result

        result["video_id"] = video_id
        self.logger.info(f"Processing Vimeo video: {video_id}")

        # Fetch video metadata
        try:
            metadata = await self._fetch_vimeo_metadata(url, video_id)
            result["metadata"] = metadata
            result["success"] = bool(metadata.get("title"))

        except Exception as e:
            self.logger.exception("Error processing Vimeo URL")
            result["error"] = f"Error processing Vimeo video: {e!s}"

        return result

    async def _fetch_vimeo_metadata(  # noqa: PLR0912
        self, url: str, video_id: str
    ) -> dict[str, Any]:
        """
        Fetch Vimeo video metadata by scraping the video page.

        Args:
            url: The Vimeo URL to fetch metadata from
            video_id: The extracted video ID (for logging purposes)

        Returns:
            Dictionary containing title, description, author, and other metadata
        """
        metadata: dict[str, Any] = {
            "title": None,
            "description": None,
            "author": None,
            "duration": None,
            "thumbnail_url": None,
            "upload_date": None,
        }

        try:
            # Use the standard Vimeo URL format for consistent access
            # Fall back to video_id-based URL if original URL is a player embed
            vimeo_url = f"https://vimeo.com/{video_id}" if "player.vimeo.com" in url else url

            response = requests.get(
                vimeo_url,
                headers=self._session_headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title from Open Graph meta tag
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                metadata["title"] = og_title["content"]

            # Extract description from meta tags
            og_description = soup.find("meta", property="og:description")
            if og_description and og_description.get("content"):
                metadata["description"] = og_description["content"]
            else:
                meta_description = soup.find("meta", attrs={"name": "description"})
                if meta_description and meta_description.get("content"):
                    metadata["description"] = meta_description["content"]

            # Extract thumbnail URL
            og_image = soup.find("meta", property="og:image")
            if og_image and og_image.get("content"):
                metadata["thumbnail_url"] = og_image["content"]

            # Try to extract additional metadata from JSON-LD
            script_tags = soup.find_all("script", type="application/ld+json")
            for script in script_tags:
                try:
                    if script.string is None:
                        continue
                    data = json.loads(script.string)
                    if isinstance(data, dict) and data.get("@type") == "VideoObject":
                        if not metadata["title"]:
                            metadata["title"] = data.get("name")
                        if not metadata["description"]:
                            metadata["description"] = data.get("description")
                        metadata["duration"] = data.get("duration")
                        metadata["upload_date"] = data.get("uploadDate")

                        # Extract author/creator
                        author_data = data.get("author") or data.get("creator")
                        if isinstance(author_data, dict):
                            metadata["author"] = author_data.get("name")
                        elif isinstance(author_data, str):
                            metadata["author"] = author_data
                except (json.JSONDecodeError, TypeError, AttributeError):
                    continue

            self.logger.debug(f"Extracted Vimeo metadata: title={metadata.get('title')}")

        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout fetching Vimeo metadata for video: {video_id}")

        except requests.exceptions.ConnectionError:
            self.logger.warning(f"Connection error fetching Vimeo metadata for video: {video_id}")

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Request error fetching Vimeo metadata: {e}")

        except Exception:
            self.logger.exception("Unexpected error fetching Vimeo metadata")

        return metadata

    # =========================================================================
    # WEBPAGE PROCESSING
    # =========================================================================

    async def process_webpage_url(self, url: str) -> dict[str, Any]:
        """
        Process a general webpage URL to extract text content.

        This method:
        1. Fetches the webpage HTML
        2. Parses with BeautifulSoup
        3. Extracts title, meta description, and main text content
        4. Filters out scripts, styles, and other non-content elements

        Args:
            url: The webpage URL to process

        Returns:
            Dictionary containing:
            - success: Boolean indicating successful processing
            - platform: "webpage"
            - url: Original URL
            - title: Page title
            - description: Meta description (if available)
            - content: Extracted text content
            - word_count: Number of words in content
            - error: Error message if processing failed

        Example:
            >>> service = URLProcessorService()
            >>> result = await service.process_webpage_url("https://example.com/article")
            >>> print(result["title"])
            "Example Article"
        """
        result: dict[str, Any] = {
            "success": False,
            "platform": "webpage",
            "url": url,
            "title": None,
            "description": None,
            "content": None,
            "word_count": 0,
            "error": None,
            "processed_at": datetime.now(tz=UTC).isoformat(),
        }

        self.logger.info(f"Processing webpage URL: {url}")

        try:
            response = requests.get(
                url,
                headers=self._session_headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            result["title"] = self._extract_webpage_title(soup)

            # Extract meta description
            result["description"] = self._extract_webpage_description(soup)

            # Extract main text content
            content = self._extract_webpage_content(soup)
            result["content"] = content
            result["word_count"] = len(content.split()) if content else 0

            result["success"] = bool(content or result["title"])

            self.logger.info(
                f"Successfully extracted webpage content: "
                f"title={result['title'][:50] if result['title'] else 'None'}..., "
                f"word_count={result['word_count']}"
            )

        except requests.exceptions.Timeout:
            result["error"] = "Request timed out while fetching webpage"
            self.logger.warning(f"Timeout fetching webpage: {url}")

        except requests.exceptions.ConnectionError:
            result["error"] = "Could not connect to the webpage"
            self.logger.warning(f"Connection error fetching webpage: {url}")

        except requests.exceptions.HTTPError as e:
            result["error"] = f"HTTP error: {e.response.status_code}"
            self.logger.warning(f"HTTP error {e.response.status_code} for: {url}")

        except requests.exceptions.RequestException as e:
            result["error"] = f"Request failed: {e!s}"
            self.logger.warning(f"Request error fetching webpage: {e}")

        except Exception as e:
            result["error"] = f"Error processing webpage: {e!s}"
            self.logger.exception("Unexpected error processing webpage")

        return result

    def _extract_webpage_title(self, soup: BeautifulSoup) -> str | None:
        """
        Extract the title from a parsed webpage.

        Tries multiple sources in order of preference:
        1. Open Graph title
        2. Twitter card title
        3. HTML title tag

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Page title string or None if not found
        """
        # Try Open Graph title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            content = og_title["content"]
            return str(content).strip() if content else None

        # Try Twitter card title
        twitter_title = soup.find("meta", attrs={"name": "twitter:title"})
        if twitter_title and twitter_title.get("content"):
            content = twitter_title["content"]
            return str(content).strip() if content else None

        # Try HTML title tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text().strip()

        return None

    def _extract_webpage_description(self, soup: BeautifulSoup) -> str | None:
        """
        Extract the description from a parsed webpage.

        Tries multiple sources in order of preference:
        1. Open Graph description
        2. Twitter card description
        3. Standard meta description

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Page description string or None if not found
        """
        # Try Open Graph description
        og_description = soup.find("meta", property="og:description")
        if og_description and og_description.get("content"):
            content = og_description["content"]
            return str(content).strip() if content else None

        # Try Twitter card description
        twitter_description = soup.find("meta", attrs={"name": "twitter:description"})
        if twitter_description and twitter_description.get("content"):
            content = twitter_description["content"]
            return str(content).strip() if content else None

        # Try standard meta description
        meta_description = soup.find("meta", attrs={"name": "description"})
        if meta_description and meta_description.get("content"):
            content = meta_description["content"]
            return str(content).strip() if content else None

        return None

    def _extract_webpage_content(self, soup: BeautifulSoup) -> str:
        """
        Extract the main text content from a parsed webpage.

        Removes scripts, styles, and other non-content elements before
        extracting visible text.

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Cleaned text content string
        """
        # Create a copy to avoid modifying the original
        soup_copy = BeautifulSoup(str(soup), "html.parser")

        # Remove non-content elements
        for element in soup_copy.find_all(
            ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]
        ):
            element.decompose()

        # Remove hidden elements
        for element in soup_copy.find_all(style=re.compile(r"display\s*:\s*none")):
            element.decompose()

        # Try to find main content areas first
        main_content = None

        # Look for common main content containers
        content_selectors = [
            soup_copy.find("main"),
            soup_copy.find("article"),
            soup_copy.find(id="content"),
            soup_copy.find(id="main-content"),
            soup_copy.find(id="main"),
            soup_copy.find(class_="content"),
            soup_copy.find(class_="post-content"),
            soup_copy.find(class_="article-content"),
            soup_copy.find(class_="entry-content"),
        ]

        for selector in content_selectors:
            if selector:
                main_content = selector
                break

        # Fall back to body if no main content area found
        if main_content is None:
            main_content = soup_copy.find("body")

        if main_content is None:
            main_content = soup_copy

        # Extract text
        text = main_content.get_text(separator=" ", strip=True)

        # Clean up text
        return self._clean_webpage_text(text)

    def _clean_webpage_text(self, text: str) -> str:
        """
        Clean extracted webpage text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text with normalized whitespace
        """
        if not text:
            return ""

        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)

        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        return text.strip()

    # =========================================================================
    # UNIVERSAL URL PROCESSOR
    # =========================================================================

    async def process_url(self, url: str) -> dict[str, Any]:
        """
        Universal URL processor that validates, detects platform, and routes
        to the appropriate processing method.

        This is the main entry point for URL processing that handles:
        1. URL validation (security checks)
        2. Platform detection (YouTube, Vimeo, or general webpage)
        3. Routing to appropriate processor
        4. Error handling and logging

        Args:
            url: The URL to process

        Returns:
            Dictionary containing processed content and metadata.
            Structure varies by platform but always includes:
            - success: Boolean indicating successful processing
            - platform: "youtube", "vimeo", or "webpage"
            - url: Original URL
            - error: Error message if processing failed

        Example:
            >>> service = URLProcessorService()
            >>> result = await service.process_url("https://www.youtube.com/watch?v=abc123")
            >>> if result["success"]:
            ...     print(f"Platform: {result['platform']}")
            ...     print(f"Content length: {len(result.get('transcript', ''))}")
        """
        # Initialize base result structure
        result: dict[str, Any] = {
            "success": False,
            "platform": None,
            "url": url,
            "error": None,
            "processed_at": datetime.now(tz=UTC).isoformat(),
        }

        # Validate URL first
        if not url:
            result["error"] = "URL is required"
            self.logger.error("process_url called with empty URL")
            return result

        url = url.strip()

        # Use centralized validation
        is_valid, url_type, error_message = validator_validate_url(url)

        if not is_valid:
            result["error"] = error_message
            self.logger.warning(f"URL validation failed: {error_message}")
            return result

        # Set platform from validation result
        result["platform"] = url_type

        self.logger.info(f"Processing URL: platform={url_type}, url={url[:100]}...")

        # Route to appropriate processor
        try:
            if url_type == "youtube":
                return await self.process_youtube_url(url)

            if url_type == "vimeo":
                return await self.process_vimeo_url(url)

            # webpage
            return await self.process_webpage_url(url)

        except Exception as e:
            result["error"] = f"Unexpected error processing URL: {e!s}"
            self.logger.error(f"Error in process_url: {e}", exc_info=True)
            return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = ["URLProcessorService"]
