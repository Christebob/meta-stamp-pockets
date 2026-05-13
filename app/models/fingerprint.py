"""
Fingerprint Pydantic model for multi-modal asset fingerprinting data.

This module defines the Fingerprint schema for storing comprehensive fingerprinting
data including perceptual hashes (pHash, aHash, dHash for images), audio spectral
analysis (mel-spectrograms, chromagrams), video frame hashes, multi-modal embeddings
(CLIP, OpenAI), and comprehensive metadata.

Supports Phase 1 fingerprint generation with Phase 2 placeholders for:
- AI training detection
- Dataset comparison
- Similarity scoring

Per Agent Action Plan section 0.6 transformation mapping and section 0.4
fingerprinting engine requirements.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Constants for validation
MIN_HASH_SIZE = 4
MAX_HASH_SIZE = 64


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


class FingerprintType(str, Enum):
    """
    Enumeration of supported fingerprint types for different media modalities.

    Each type corresponds to a specific content category that requires
    different fingerprinting algorithms and metadata extraction strategies.
    """

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"
    URL = "url"


class ProcessingStatus(str, Enum):
    """
    Enumeration of fingerprint processing states.

    Tracks the lifecycle of fingerprint generation from initial queueing
    through processing to completion or failure.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Fingerprint(BaseModel):
    """
    Comprehensive Pydantic model for multi-modal asset fingerprinting data.

    This model stores all fingerprint-related data for creative assets including:
    - Perceptual hashes for images (pHash, aHash, dHash)
    - Audio spectral fingerprints (mel-spectrogram, chromagram, spectral centroid)
    - Video frame hashes with frame sampling metadata
    - Multi-modal embeddings from LangChain/OpenAI/CLIP
    - Comprehensive metadata extraction results
    - Phase 2 placeholders for AI training detection fields

    The schema is flexible to accommodate different media types, with Optional
    fields allowing storage of type-specific data without requiring all fields
    for every fingerprint.

    Attributes:
        id: MongoDB ObjectId as string (aliased from _id)
        asset_id: Unique reference to the associated Asset document
        user_id: Reference to the user who owns the asset
        fingerprint_type: Type of content being fingerprinted
        perceptual_hashes: Image hash data (pHash, aHash, dHash)
        spectral_data: Audio spectral analysis data
        video_hashes: Video frame hash data
        embeddings: Multi-modal embedding vectors
        image_metadata: Image-specific metadata (EXIF, dimensions)
        audio_metadata: Audio-specific metadata (codec, bitrate)
        video_metadata: Video-specific metadata (resolution, fps)
        text_hash: Hash of text content
        text_length: Length of text content in characters
        url_metadata: URL-specific metadata
        training_detected: Phase 2 - AI training detection result
        dataset_matches: Phase 2 - Matched AI training datasets
        similarity_scores: Phase 2 - Similarity scores against datasets
        legal_status: Phase 2 - Legal determination status
        processing_status: Current processing state
        error_message: Error details if processing failed
        processing_duration: Time taken to generate fingerprint
        created_at: Timestamp when fingerprint was created
        updated_at: Timestamp when fingerprint was last modified
    """

    # MongoDB Configuration
    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={datetime: lambda v: v.isoformat()},
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    # Core Fingerprint Fields
    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")

    asset_id: str = Field(
        ...,
        description="Unique reference to the associated Asset document. "
        "Establishes 1:1 relationship between Asset and Fingerprint.",
        min_length=1,
    )

    user_id: str = Field(..., description="Reference to the user who owns the asset", min_length=1)

    fingerprint_type: FingerprintType = Field(
        ..., description="Type of content being fingerprinted (image, audio, video, text, url)"
    )

    # Image Fingerprinting Data
    perceptual_hashes: dict[str, Any] | None = Field(
        default=None,
        description="Perceptual hash data for images containing:\n"
        "- phash: perceptual hash using DCT (string, hex format)\n"
        "- ahash: average hash (string, hex format)\n"
        "- dhash: difference hash (string, hex format)\n"
        "- hash_size: hash bit length (int, typically 8 or 16)",
    )

    image_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Image-specific metadata including:\n"
        "- width: image width in pixels\n"
        "- height: image height in pixels\n"
        "- format: image format (PNG, JPEG, WebP)\n"
        "- color_mode: color mode (RGB, RGBA, L)\n"
        "- exif: EXIF metadata dictionary",
    )

    # Audio Fingerprinting Data
    spectral_data: dict[str, Any] | None = Field(
        default=None,
        description="Audio spectral analysis data containing:\n"
        "- mel_spectrogram: frequency representation (List[float])\n"
        "- chromagram: pitch class representation (List[float])\n"
        "- spectral_centroid: brightness measure (float)\n"
        "- duration: duration in seconds (float)\n"
        "- sample_rate: audio sample rate (int)",
    )

    audio_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Audio-specific metadata including:\n"
        "- codec: audio codec (e.g., mp3, wav, aac)\n"
        "- bitrate: audio bitrate in kbps\n"
        "- channels: number of audio channels\n"
        "- duration: duration in seconds",
    )

    # Video Fingerprinting Data
    video_hashes: dict[str, Any] | None = Field(
        default=None,
        description="Video frame hash data containing:\n"
        "- frame_hashes: list of representative frame hashes (List[str])\n"
        "- sampling_interval: frame extraction interval in seconds (float)\n"
        "- total_frames_analyzed: number of frames processed (int)\n"
        "- average_hash: overall video hash (str)",
    )

    video_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Video-specific metadata including:\n"
        "- width: video width in pixels\n"
        "- height: video height in pixels\n"
        "- codec: video codec (e.g., h264, hevc)\n"
        "- duration: duration in seconds\n"
        "- fps: frames per second",
    )

    # Multi-Modal Embeddings
    embeddings: dict[str, Any] | None = Field(
        default=None,
        description="Multi-modal embedding vectors containing:\n"
        "- clip_embedding: 512-dimensional CLIP vector (List[float])\n"
        "- openai_embedding: 1536-dimensional OpenAI vector (List[float])\n"
        "- embedding_model: model identifier (str)\n"
        "- embedding_version: version tracking (str)",
    )

    # Text/URL Fingerprinting Data
    text_hash: str | None = Field(
        default=None, description="SHA-256 hash of text content for quick comparison"
    )

    text_length: int | None = Field(
        default=None, ge=0, description="Length of text content in characters"
    )

    url_metadata: dict[str, Any] | None = Field(
        default=None,
        description="URL-specific metadata including:\n"
        "- original_url: original URL submitted\n"
        "- platform: detected platform (youtube, vimeo, web)\n"
        "- title: page/video title\n"
        "- description: page/video description\n"
        "- thumbnail_url: thumbnail image URL\n"
        "- transcript: extracted transcript (for videos)\n"
        "- fetch_timestamp: when URL was fetched",
    )

    # Phase 2 Placeholder Fields
    # TODO Phase 2: Implement AI training detection engine
    # These fields will store results from comparing fingerprints against
    # known AI training datasets to detect unauthorized use of creator content
    training_detected: bool | None = Field(
        default=None,
        description="Phase 2: Boolean indicating if asset was detected in AI training data. "
        "TODO Phase 2: Implement AI training detection engine",
    )

    # TODO Phase 2: Compare embeddings against known datasets
    dataset_matches: list[str] | None = Field(
        default=None,
        description="Phase 2: List of dataset identifiers where asset was found. "
        "TODO Phase 2: Compare embeddings against known datasets",
    )

    # TODO Phase 2: Calculate embedding drift scores
    similarity_scores: dict[str, float] | None = Field(
        default=None,
        description="Phase 2: Similarity scores against matched datasets (0.0-1.0). "
        "TODO Phase 2: Calculate embedding drift scores and similarity metrics",
    )

    # TODO Phase 2: Apply similarity-law thresholds for legal determination
    legal_status: str | None = Field(
        default=None,
        description="Phase 2: Legal status determination based on similarity analysis. "
        "TODO Phase 2: Apply similarity-law thresholds for legal determination",
    )

    # Processing Status Fields
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing state of the fingerprint generation",
    )

    error_message: str | None = Field(
        default=None, description="Error details if fingerprint processing failed"
    )

    processing_duration: float | None = Field(
        default=None, ge=0.0, description="Time taken to generate fingerprint in seconds"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=_utc_now, description="Timestamp when fingerprint was created (UTC)"
    )

    updated_at: datetime = Field(
        default_factory=_utc_now, description="Timestamp when fingerprint was last modified (UTC)"
    )

    # Validators
    @field_validator("perceptual_hashes")
    @classmethod
    def validate_perceptual_hashes(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Validate perceptual hash data structure and format.

        Ensures hash strings are in valid hexadecimal format and
        hash_size is a reasonable value for image hashing.

        Args:
            v: Dictionary containing hash data or None

        Returns:
            Validated hash data dictionary or None

        Raises:
            ValueError: If hash format is invalid
        """
        if v is None:
            return v

        # Validate hash string formats if present
        hash_keys = ["phash", "ahash", "dhash"]
        for key in hash_keys:
            if key in v and v[key] is not None:
                hash_value = v[key]
                if isinstance(hash_value, str):
                    # Validate hexadecimal format
                    try:
                        int(hash_value, 16)
                    except ValueError as err:
                        raise ValueError(
                            f"Invalid {key} format: must be hexadecimal string, got '{hash_value}'"
                        ) from err

        # Validate hash_size if present
        if "hash_size" in v and v["hash_size"] is not None:
            hash_size = v["hash_size"]
            if (
                not isinstance(hash_size, int)
                or hash_size < MIN_HASH_SIZE
                or hash_size > MAX_HASH_SIZE
            ):
                raise ValueError(
                    f"Invalid hash_size: must be integer between {MIN_HASH_SIZE} and {MAX_HASH_SIZE}, got {hash_size}"
                )

        return v

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Validate embedding vectors and their dimensions.

        Ensures embedding vectors have expected dimensions:
        - CLIP embeddings: 512 dimensions
        - OpenAI embeddings: 1536 dimensions

        Args:
            v: Dictionary containing embedding data or None

        Returns:
            Validated embedding data dictionary or None

        Raises:
            ValueError: If embedding dimensions are invalid
        """
        if v is None:
            return v

        # Expected dimensions for different embedding models
        expected_dimensions = {
            "clip_embedding": 512,
            "openai_embedding": 1536,
        }

        for key, expected_dim in expected_dimensions.items():
            if key in v and v[key] is not None:
                embedding = v[key]
                if isinstance(embedding, list):
                    actual_dim = len(embedding)
                    # Allow some flexibility for different model versions
                    # but warn if significantly different
                    if actual_dim > 0 and abs(actual_dim - expected_dim) > expected_dim * 0.5:
                        # Only raise for extreme deviations
                        raise ValueError(
                            f"Invalid {key} dimensions: expected approximately {expected_dim}, "
                            f"got {actual_dim}"
                        )

        return v

    @field_validator("spectral_data")
    @classmethod
    def validate_spectral_data(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Validate audio spectral analysis data structure.

        Ensures spectral data contains valid numeric values for
        audio fingerprinting fields.

        Args:
            v: Dictionary containing spectral data or None

        Returns:
            Validated spectral data dictionary or None

        Raises:
            ValueError: If spectral data format is invalid
        """
        if v is None:
            return v

        # Validate duration is positive
        if "duration" in v and v["duration"] is not None:
            duration = v["duration"]
            if not isinstance(duration, (int, float)) or duration < 0:
                raise ValueError(f"Invalid duration: must be non-negative number, got {duration}")

        # Validate sample_rate is positive integer
        if "sample_rate" in v and v["sample_rate"] is not None:
            sample_rate = v["sample_rate"]
            if not isinstance(sample_rate, int) or sample_rate <= 0:
                raise ValueError(
                    f"Invalid sample_rate: must be positive integer, got {sample_rate}"
                )

        # Validate spectral_centroid is non-negative
        if "spectral_centroid" in v and v["spectral_centroid"] is not None:
            centroid = v["spectral_centroid"]
            if not isinstance(centroid, (int, float)) or centroid < 0:
                raise ValueError(
                    f"Invalid spectral_centroid: must be non-negative number, got {centroid}"
                )

        return v

    @field_validator("video_hashes")
    @classmethod
    def validate_video_hashes(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Validate video frame hash data structure.

        Ensures video hash data contains valid frame hashes and
        sampling metadata.

        Args:
            v: Dictionary containing video hash data or None

        Returns:
            Validated video hash data dictionary or None

        Raises:
            ValueError: If video hash format is invalid
        """
        if v is None:
            return v

        # Validate sampling_interval is positive
        if "sampling_interval" in v and v["sampling_interval"] is not None:
            interval = v["sampling_interval"]
            if not isinstance(interval, (int, float)) or interval <= 0:
                raise ValueError(
                    f"Invalid sampling_interval: must be positive number, got {interval}"
                )

        # Validate total_frames_analyzed is non-negative integer
        if "total_frames_analyzed" in v and v["total_frames_analyzed"] is not None:
            total_frames = v["total_frames_analyzed"]
            if not isinstance(total_frames, int) or total_frames < 0:
                raise ValueError(
                    f"Invalid total_frames_analyzed: must be non-negative integer, "
                    f"got {total_frames}"
                )

        # Validate frame_hashes is a list of strings
        if "frame_hashes" in v and v["frame_hashes"] is not None:
            frame_hashes = v["frame_hashes"]
            if not isinstance(frame_hashes, list):
                raise TypeError(
                    f"Invalid frame_hashes: must be list, got {type(frame_hashes).__name__}"
                )
            for i, hash_value in enumerate(frame_hashes):
                if not isinstance(hash_value, str):
                    raise TypeError(
                        f"Invalid frame_hash at index {i}: must be string, "
                        f"got {type(hash_value).__name__}"
                    )

        return v

    @field_validator("similarity_scores")
    @classmethod
    def validate_similarity_scores(cls, v: dict[str, float] | None) -> dict[str, float] | None:
        """
        Validate similarity scores are within valid range (0.0-1.0).

        Phase 2 field for storing similarity comparison results against
        known AI training datasets.

        Args:
            v: Dictionary of dataset IDs to similarity scores or None

        Returns:
            Validated similarity scores dictionary or None

        Raises:
            ValueError: If scores are outside valid range
        """
        if v is None:
            return v

        for dataset_id, score in v.items():
            if not isinstance(score, (int, float)):
                raise TypeError(
                    f"Invalid similarity score for '{dataset_id}': must be number, "
                    f"got {type(score).__name__}"
                )
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    f"Invalid similarity score for '{dataset_id}': must be between "
                    f"0.0 and 1.0, got {score}"
                )

        return v

    def mark_processing(self) -> None:
        """
        Mark fingerprint as currently processing.

        Updates processing_status to PROCESSING and refreshes updated_at timestamp.
        """
        self.processing_status = ProcessingStatus.PROCESSING
        self.updated_at = datetime.now(UTC)

    def mark_completed(self, duration: float | None = None) -> None:
        """
        Mark fingerprint processing as completed successfully.

        Updates processing_status to COMPLETED, optionally records duration,
        and refreshes updated_at timestamp.

        Args:
            duration: Optional processing duration in seconds
        """
        self.processing_status = ProcessingStatus.COMPLETED
        if duration is not None:
            self.processing_duration = duration
        self.error_message = None
        self.updated_at = datetime.now(UTC)

    def mark_failed(self, error_message: str, duration: float | None = None) -> None:
        """
        Mark fingerprint processing as failed.

        Updates processing_status to FAILED, records error message,
        optionally records duration, and refreshes updated_at timestamp.

        Args:
            error_message: Description of the error that occurred
            duration: Optional processing duration in seconds before failure
        """
        self.processing_status = ProcessingStatus.FAILED
        self.error_message = error_message
        if duration is not None:
            self.processing_duration = duration
        self.updated_at = datetime.now(UTC)

    def is_complete(self) -> bool:
        """
        Check if fingerprint processing is complete.

        Returns:
            True if processing status is COMPLETED, False otherwise
        """
        return self.processing_status == ProcessingStatus.COMPLETED

    def has_image_fingerprint(self) -> bool:
        """
        Check if fingerprint contains image-specific data.

        Returns:
            True if perceptual_hashes is populated, False otherwise
        """
        return self.perceptual_hashes is not None and len(self.perceptual_hashes) > 0

    def has_audio_fingerprint(self) -> bool:
        """
        Check if fingerprint contains audio-specific data.

        Returns:
            True if spectral_data is populated, False otherwise
        """
        return self.spectral_data is not None and len(self.spectral_data) > 0

    def has_video_fingerprint(self) -> bool:
        """
        Check if fingerprint contains video-specific data.

        Returns:
            True if video_hashes is populated, False otherwise
        """
        return self.video_hashes is not None and len(self.video_hashes) > 0

    def has_embeddings(self) -> bool:
        """
        Check if fingerprint contains embedding vectors.

        Returns:
            True if embeddings is populated, False otherwise
        """
        return self.embeddings is not None and len(self.embeddings) > 0

    def to_mongodb_dict(self) -> dict[str, Any]:
        """
        Convert model to dictionary suitable for MongoDB insertion.

        Handles _id field aliasing and datetime serialization for
        proper MongoDB document storage.

        Returns:
            Dictionary representation suitable for MongoDB
        """
        data = self.model_dump(by_alias=True, exclude_none=False)

        # Remove _id if None (let MongoDB generate it)
        if data.get("_id") is None:
            data.pop("_id", None)

        return data

    @classmethod
    def from_mongodb_dict(cls, data: dict[str, Any]) -> "Fingerprint":
        """
        Create Fingerprint instance from MongoDB document.

        Handles _id field conversion from ObjectId to string.

        Args:
            data: MongoDB document dictionary

        Returns:
            Fingerprint model instance
        """
        # Convert ObjectId to string if present
        if "_id" in data and data["_id"] is not None:
            data["_id"] = str(data["_id"])

        return cls.model_validate(data)
