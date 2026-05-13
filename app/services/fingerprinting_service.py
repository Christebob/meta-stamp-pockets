"""
Multi-modal Fingerprinting Service for META-STAMP V3

This module provides comprehensive multi-modal fingerprinting for creative assets,
generating unique identifiers using:
- Perceptual hashing (imagehash pHash/aHash/dHash) for images
- Spectral analysis (librosa mel-spectrogram/chromagram) for audio
- Frame-based hashing (opencv with 1-second interval sampling) for video
- Semantic embeddings (LangChain OpenAI) for multi-modal content

Includes Phase 2 TODO markers for:
- AI training detection engine
- Dataset comparison against known AI training datasets
- Embedding drift analysis
- Similarity-law thresholds for legal determination
- Legal export documentation generation

Per Agent Action Plan:
- Section 0.3: Phase 2 preparation with TODO markers (MANDATORY)
- Section 0.4: Multi-modal fingerprinting implementation
- Section 0.6: FingerprintingService requirements
- Section 0.8: Fingerprinting algorithms using imagehash, librosa, opencv

Author: META-STAMP V3 Platform
License: Proprietary
"""

import hashlib
import logging
import tempfile
import time

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import imagehash
import librosa
import numpy as np

from bson import ObjectId
# langchain_openai is imported lazily inside __init__ and factory functions
# to avoid 2-3s startup delay from OpenAI SDK eager loading.
from PIL import Image, UnidentifiedImageError
from pydantic import SecretStr

from app.core.database import get_db_client
from app.models.fingerprint import FingerprintType, ProcessingStatus
from app.services.metadata_service import MetadataService
from app.services.storage_service import StorageService

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency/provider import
    OpenAIEmbeddings = None  # type: ignore[assignment]


# Configure module logger for tracking fingerprinting operations
logger = logging.getLogger(__name__)

# Constants for fingerprinting
DEFAULT_HASH_SIZE = 16  # Hash size for perceptual hashing (produces 16x16 = 256-bit hash)
AUDIO_SAMPLE_RATE = 22050  # Standard sample rate for audio analysis
VIDEO_FRAME_INTERVAL = 1.0  # Extract frames every 1 second
MAX_VIDEO_FRAMES = 300  # Maximum frames to analyze per video (5 minutes at 1fps)
MAX_EMBEDDING_CONTENT_LENGTH = 8000  # Maximum characters for embedding generation
MEL_SPECTROGRAM_N_MELS = 128  # Number of mel bands for spectrogram
CHROMA_N_CHROMA = 12  # Number of chroma bins (standard chromatic scale)


class FingerprintingServiceError(Exception):
    """Base exception for fingerprinting service errors."""


class FingerprintGenerationError(FingerprintingServiceError):
    """Raised when fingerprint generation fails."""


class UnsupportedFileTypeError(FingerprintingServiceError):
    """Raised when file type is not supported for fingerprinting."""


class FingerprintingService:
    """
    Multi-modal fingerprinting service for creative asset identification.

    This service generates unique fingerprints for various media types using:
    - Image fingerprinting: Perceptual hashing (pHash, aHash, dHash) resistant
      to minor modifications using discrete cosine transform
    - Audio fingerprinting: Spectral analysis with mel-spectrograms, chromagrams,
      and spectral centroids for capturing timbral characteristics
    - Video fingerprinting: Frame extraction at 1-second intervals with image
      hashing for representative frames
    - Text fingerprinting: SHA-256 content hashing with embedding generation
    - Semantic embeddings: LangChain OpenAI embeddings for multi-modal similarity

    The service integrates with:
    - StorageService: Download files from S3/MinIO for local processing
    - MetadataService: Extract comprehensive file metadata
    - MongoDB: Store fingerprint records via get_db_client()

    Attributes:
        storage: StorageService instance for S3 file operations
        metadata: MetadataService instance for metadata extraction
        logger: Logger instance for operation tracking
        embeddings: OpenAIEmbeddings instance for embedding generation (optional)

    Example:
        >>> storage = StorageService(bucket_name="assets", endpoint_url="http://minio:9000")
        >>> metadata = MetadataService()
        >>> service = FingerprintingService(storage, metadata, openai_api_key="your-openai-api-key")
        >>> fingerprint = await service.generate_fingerprint(
        ...     asset_id="abc123",
        ...     object_key="uploads/image.jpg",
        ...     file_type="image",
        ...     user_id="user123"
        ... )
    """

    def __init__(
        self,
        storage_service: StorageService,
        metadata_service: MetadataService,
        openai_api_key: str | None = None,
    ) -> None:
        """
        Initialize the FingerprintingService with required dependencies.

        Args:
            storage_service: StorageService instance for downloading files from S3/MinIO
            metadata_service: MetadataService instance for extracting file metadata
            openai_api_key: Optional OpenAI API key for embedding generation.
                           If None, embeddings will not be generated.
        """
        self.storage = storage_service
        self.metadata = metadata_service
        self.logger = logging.getLogger(__name__)
        self.embeddings: Any | None = None

        # Initialize OpenAI embeddings if API key provided
        if openai_api_key:
            try:
                if OpenAIEmbeddings is None:
                    raise RuntimeError("langchain_openai.OpenAIEmbeddings is unavailable")
                self.embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
                self.logger.info("OpenAI embeddings initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
                self.embeddings = None
        else:
            self.logger.info("OpenAI embeddings not configured (no API key provided)")

    async def fingerprint_image(self, file_path: str) -> dict[str, Any]:
        """
        Generate comprehensive fingerprint for image files.

        Loads the image using PIL and generates multiple perceptual hashes that
        are resistant to minor image modifications like compression, scaling,
        and color adjustments:

        - pHash (Perceptual Hash): Uses discrete cosine transform (DCT) to capture
          frequency domain features, most robust against modifications
        - aHash (Average Hash): Simple and fast, compares pixels to mean value
        - dHash (Difference Hash): Captures gradient information, good for scaling

        Also extracts EXIF metadata and generates OpenAI embedding if configured.

        Args:
            file_path: Absolute path to the image file on local filesystem

        Returns:
            Dictionary containing:
                - perceptual_hashes: dict with phash, ahash, dhash, hash_size
                - metadata: EXIF and image properties from MetadataService
                - embeddings: dict with openai_embedding if available
                - fingerprint_type: "image"

        Raises:
            FingerprintGenerationError: If image cannot be loaded or processed
            FileNotFoundError: If the file does not exist

        Example:
            >>> result = await service.fingerprint_image("/tmp/photo.jpg")
            >>> print(result["perceptual_hashes"]["phash"])  # "a7f39c8e..."
        """
        self.logger.info(f"Generating image fingerprint for: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"Image file not found: {file_path}")
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            # Load image with PIL
            with Image.open(file_path) as img:
                # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
                img_converted = img.convert("RGB") if img.mode not in ("RGB", "L") else img

                # Resize to standard dimensions for consistent hashing
                # This ensures same hash regardless of original resolution
                img_resized = img_converted.resize((256, 256), Image.Resampling.LANCZOS)

                # Generate perceptual hashes using imagehash library
                # pHash: Uses DCT (discrete cosine transform) - most robust
                phash = imagehash.phash(img_resized, hash_size=DEFAULT_HASH_SIZE)

                # aHash: Average hash - compares each pixel to mean
                ahash = imagehash.average_hash(img_resized, hash_size=DEFAULT_HASH_SIZE)

                # dHash: Difference hash - captures gradient information
                dhash = imagehash.dhash(img_resized, hash_size=DEFAULT_HASH_SIZE)

                phash_str = self._normalize_hash(phash)
                ahash_str = self._normalize_hash(ahash)
                dhash_str = self._normalize_hash(dhash)

                perceptual_hashes: dict[str, str | int] = {
                    "phash": phash_str,
                    "ahash": ahash_str,
                    "dhash": dhash_str,
                    "hash_size": DEFAULT_HASH_SIZE,
                }

                self.logger.debug(
                    f"Generated hashes - pHash: {phash_str[:16]}..., "
                    f"aHash: {ahash_str[:16]}..., "
                    f"dHash: {dhash_str[:16]}..."
                )

            # Extract comprehensive image metadata
            image_metadata = await self.metadata.extract_image_metadata(file_path)

            # Generate embedding if OpenAI is configured
            embedding_data: dict[str, Any] = {}
            if self.embeddings:
                # Create text description for embedding
                description = (
                    f"Image file: {path.name}, "
                    f"dimensions: {image_metadata.get('width', 'unknown')}x"
                    f"{image_metadata.get('height', 'unknown')}, "
                    f"format: {image_metadata.get('format', 'unknown')}"
                )
                openai_embedding = await self._generate_embedding(description)
                if openai_embedding:
                    embedding_data = {
                        "openai_embedding": openai_embedding,
                        "embedding_model": "text-embedding-ada-002",
                        "embedding_version": "1.0",
                    }

            result = {
                "perceptual_hashes": perceptual_hashes,
                "image_metadata": image_metadata,
                "embeddings": embedding_data if embedding_data else None,
                "fingerprint_type": FingerprintType.IMAGE.value,
            }

            self.logger.info(f"Image fingerprint generated successfully for {path.name}")
            return result

        except UnidentifiedImageError as e:
            error_msg = f"Cannot identify image file: {file_path}"
            self.logger.exception(error_msg)
            raise FingerprintGenerationError(error_msg) from e
        except Exception as e:
            error_msg = f"Error generating image fingerprint: {e}"
            self.logger.exception(error_msg)
            raise FingerprintGenerationError(error_msg) from e

    async def fingerprint_audio(self, file_path: str) -> dict[str, Any]:
        """
        Generate comprehensive fingerprint for audio files.

        Loads the audio file using librosa and extracts spectral features that
        capture the unique timbral characteristics of the audio:

        - Mel-spectrogram: Frequency-time representation using mel scale
          for human perception, captures tonal structure
        - Chromagram: Pitch class distribution (12 semitones) capturing
          harmonic content independent of octave
        - Spectral centroid: Brightness measure indicating where the
          "center of mass" of the spectrum is located

        Also generates a hash of the spectral features and OpenAI embedding.

        Args:
            file_path: Absolute path to the audio file on local filesystem

        Returns:
            Dictionary containing:
                - spectral_data: dict with mel_spectrogram_hash, chromagram_hash,
                  spectral_centroid_mean, duration, sample_rate
                - audio_metadata: Properties from MetadataService
                - embeddings: dict with openai_embedding if available
                - fingerprint_type: "audio"

        Raises:
            FingerprintGenerationError: If audio cannot be loaded or processed
            FileNotFoundError: If the file does not exist

        Example:
            >>> result = await service.fingerprint_audio("/tmp/song.mp3")
            >>> print(result["spectral_data"]["spectral_centroid_mean"])  # 3500.25
        """
        self.logger.info(f"Generating audio fingerprint for: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"Audio file not found: {file_path}")
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            # Load audio with librosa at consistent sample rate
            # sr=AUDIO_SAMPLE_RATE ensures consistent analysis
            audio_data, sample_rate = librosa.load(file_path, sr=AUDIO_SAMPLE_RATE, mono=True)

            # Get audio duration
            duration = librosa.get_duration(y=audio_data, sr=sample_rate)

            # Extract mel-spectrogram (frequency-time representation)
            # Uses mel scale which approximates human auditory perception
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=MEL_SPECTROGRAM_N_MELS,
                fmax=8000,  # Focus on frequencies most relevant to human perception
            )

            # Convert to log scale for better representation
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Compute hash of mel-spectrogram for quick comparison
            mel_hash = self._compute_array_hash(mel_spectrogram_db)

            # Extract chromagram (pitch class distribution)
            # Captures harmonic content in 12 pitch classes
            chromagram = librosa.feature.chroma_stft(
                y=audio_data, sr=sample_rate, n_chroma=CHROMA_N_CHROMA
            )

            # Compute hash of chromagram
            chroma_hash = self._compute_array_hash(chromagram)

            # Calculate spectral centroid (timbral brightness measure)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]

            # Compute statistics for fingerprinting
            spectral_centroid_mean = float(np.mean(spectral_centroids))
            spectral_centroid_std = float(np.std(spectral_centroids))

            spectral_data = {
                "mel_spectrogram_hash": mel_hash,
                "chromagram_hash": chroma_hash,
                "spectral_centroid_mean": round(spectral_centroid_mean, 4),
                "spectral_centroid_std": round(spectral_centroid_std, 4),
                "duration": round(duration, 3),
                "sample_rate": sample_rate,
                "n_mels": MEL_SPECTROGRAM_N_MELS,
                "n_chroma": CHROMA_N_CHROMA,
            }

            self.logger.debug(
                f"Generated spectral data - duration: {duration:.2f}s, "
                f"centroid: {spectral_centroid_mean:.2f}Hz"
            )

            # Extract comprehensive audio metadata
            audio_metadata = await self.metadata.extract_audio_metadata(file_path)

            # Generate embedding if OpenAI is configured
            embedding_data: dict[str, Any] = {}
            if self.embeddings:
                description = (
                    f"Audio file: {path.name}, "
                    f"duration: {audio_metadata.get('duration_formatted', 'unknown')}, "
                    f"sample_rate: {sample_rate}Hz, "
                    f"spectral_centroid: {spectral_centroid_mean:.2f}Hz"
                )
                openai_embedding = await self._generate_embedding(description)
                if openai_embedding:
                    embedding_data = {
                        "openai_embedding": openai_embedding,
                        "embedding_model": "text-embedding-ada-002",
                        "embedding_version": "1.0",
                    }

            result = {
                "spectral_data": spectral_data,
                "audio_metadata": audio_metadata,
                "embeddings": embedding_data if embedding_data else None,
                "fingerprint_type": FingerprintType.AUDIO.value,
            }

            self.logger.info(f"Audio fingerprint generated successfully for {path.name}")
            return result

        except Exception as e:
            error_msg = f"Error generating audio fingerprint: {e}"
            self.logger.exception(error_msg)
            raise FingerprintGenerationError(error_msg) from e

    async def fingerprint_video(self, file_path: str) -> dict[str, Any]:
        """
        Generate comprehensive fingerprint for video files.

        Opens the video using OpenCV VideoCapture and extracts frames at
        1-second intervals. For each frame, generates perceptual hashes
        using the same approach as image fingerprinting. Computes an
        average hash across all frames for overall video identification.

        Args:
            file_path: Absolute path to the video file on local filesystem

        Returns:
            Dictionary containing:
                - video_hashes: dict with frame_hashes (list), average_hash,
                  sampling_interval, total_frames_analyzed
                - video_metadata: Properties from MetadataService
                - embeddings: dict with openai_embedding if available
                - fingerprint_type: "video"

        Raises:
            FingerprintGenerationError: If video cannot be loaded or processed
            FileNotFoundError: If the file does not exist

        Example:
            >>> result = await service.fingerprint_video("/tmp/video.mp4")
            >>> print(len(result["video_hashes"]["frame_hashes"]))  # Number of analyzed frames
        """
        self.logger.info(f"Generating video fingerprint for: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"Video file not found: {file_path}")
            raise FileNotFoundError(f"Video file not found: {file_path}")

        cap = None
        try:
            # Open video with OpenCV VideoCapture
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                raise FingerprintGenerationError(f"Cannot open video file: {file_path}")

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps <= 0:
                fps = 30.0  # Default to 30 fps if unknown
                self.logger.warning(f"Unknown FPS for {file_path}, using default {fps}")

            # Calculate frame interval for 1-second sampling
            frame_interval = int(fps * VIDEO_FRAME_INTERVAL)
            frame_interval = max(frame_interval, 1)

            self.logger.debug(
                f"Video analysis: {total_frames} frames @ {fps:.2f}fps, "
                f"sampling every {frame_interval} frames"
            )

            # Extract frames and generate hashes using helper method
            frame_hashes, frames_analyzed = self._extract_video_frame_hashes(cap, frame_interval)

            # Compute average hash across all frames if we have any
            average_hash = ""
            if frame_hashes:
                # Convert hex hashes to integers for averaging
                hash_values = [int(h, 16) for h in frame_hashes]
                avg_value = int(np.mean(hash_values))
                average_hash = format(avg_value, f"0{DEFAULT_HASH_SIZE * 4}x")

            video_hashes = {
                "frame_hashes": frame_hashes,
                "average_hash": average_hash,
                "sampling_interval": VIDEO_FRAME_INTERVAL,
                "total_frames_analyzed": frames_analyzed,
                "total_video_frames": total_frames,
                "fps": round(fps, 2),
            }

            self.logger.debug(
                f"Generated video hashes - {frames_analyzed} frames analyzed, "
                f"average_hash: {average_hash[:16] if average_hash else 'N/A'}..."
            )

            # Extract comprehensive video metadata
            video_metadata = await self.metadata.extract_video_metadata(file_path)

            # Generate embedding if OpenAI is configured
            embedding_data: dict[str, Any] = {}
            if self.embeddings:
                description = (
                    f"Video file: {path.name}, "
                    f"resolution: {video_metadata.get('resolution', 'unknown')}, "
                    f"duration: {video_metadata.get('duration_formatted', 'unknown')}, "
                    f"frames_analyzed: {frames_analyzed}"
                )
                openai_embedding = await self._generate_embedding(description)
                if openai_embedding:
                    embedding_data = {
                        "openai_embedding": openai_embedding,
                        "embedding_model": "text-embedding-ada-002",
                        "embedding_version": "1.0",
                    }

            result = {
                "video_hashes": video_hashes,
                "video_metadata": video_metadata,
                "embeddings": embedding_data if embedding_data else None,
                "fingerprint_type": FingerprintType.VIDEO.value,
            }

            self.logger.info(f"Video fingerprint generated successfully for {path.name}")
            return result

        except cv2.error as e:
            error_msg = f"OpenCV error processing video: {e}"
            self.logger.exception(error_msg)
            raise FingerprintGenerationError(error_msg) from e
        except Exception as e:
            error_msg = f"Error generating video fingerprint: {e}"
            self.logger.exception(error_msg)
            raise FingerprintGenerationError(error_msg) from e
        finally:
            # Always release video capture resources
            if cap is not None:
                cap.release()

    async def fingerprint_text(self, content: str) -> dict[str, Any]:
        """
        Generate fingerprint for text content.

        Computes SHA-256 hash of the text content for quick comparison,
        extracts basic statistics, and generates OpenAI embedding for
        semantic similarity detection.

        Args:
            content: Text content to fingerprint

        Returns:
            Dictionary containing:
                - text_hash: SHA-256 hash of content
                - text_length: Character count
                - word_count: Approximate word count
                - line_count: Number of lines
                - embeddings: dict with openai_embedding if available
                - fingerprint_type: "text"

        Example:
            >>> result = await service.fingerprint_text("Hello, World!")
            >>> print(result["text_hash"])  # SHA-256 hash
        """
        self.logger.info(f"Generating text fingerprint for content ({len(content)} chars)")

        try:
            # Normalize text for consistent hashing
            normalized_content = content.strip()

            # Compute SHA-256 hash of content
            text_hash = hashlib.sha256(normalized_content.encode("utf-8")).hexdigest()

            # Extract basic statistics
            text_length = len(normalized_content)
            word_count = len(normalized_content.split())
            line_count = len(normalized_content.splitlines())

            # Generate embedding if OpenAI is configured
            embedding_data: dict[str, Any] = {}
            if self.embeddings and normalized_content:
                # Use actual content for embedding (truncated if too long)
                embedding_content = (
                    normalized_content[:MAX_EMBEDDING_CONTENT_LENGTH]
                    if len(normalized_content) > MAX_EMBEDDING_CONTENT_LENGTH
                    else normalized_content
                )
                openai_embedding = await self._generate_embedding(embedding_content)
                if openai_embedding:
                    embedding_data = {
                        "openai_embedding": openai_embedding,
                        "embedding_model": "text-embedding-ada-002",
                        "embedding_version": "1.0",
                    }

            result = {
                "text_hash": text_hash,
                "text_length": text_length,
                "word_count": word_count,
                "line_count": line_count,
                "embeddings": embedding_data if embedding_data else None,
                "fingerprint_type": FingerprintType.TEXT.value,
            }

            self.logger.info(
                f"Text fingerprint generated - hash: {text_hash[:16]}..., "
                f"{word_count} words, {line_count} lines"
            )
            return result

        except Exception as e:
            error_msg = f"Error generating text fingerprint: {e}"
            self.logger.exception(error_msg)
            raise FingerprintGenerationError(error_msg) from e

    async def generate_fingerprint(
        self,
        asset_id: str,
        object_key: str,
        file_type: str,
        user_id: str,
    ) -> dict[str, Any]:
        """
        Universal fingerprint generator that downloads file and routes to appropriate method.

        This is the main entry point for fingerprint generation. It:
        1. Downloads the file from S3/MinIO to a temporary location
        2. Routes to the appropriate fingerprinting method based on file_type
        3. Creates a Fingerprint record in MongoDB with asset_id reference
        4. Cleans up the temporary file
        5. Returns the complete fingerprint data

        Args:
            asset_id: Unique reference to the associated Asset document
            object_key: S3 object key (path) for the file in storage
            file_type: Type of file ("image", "audio", "video", "text")
            user_id: Reference to the user who owns the asset

        Returns:
            Dictionary containing complete fingerprint data including:
                - fingerprint_id: MongoDB ObjectId as string
                - asset_id: Reference to associated asset
                - fingerprint_type: Type of fingerprint generated
                - processing_status: "completed" on success
                - processing_duration: Time taken in seconds
                - ... type-specific fingerprint data

        Raises:
            UnsupportedFileTypeError: If file_type is not supported
            FingerprintGenerationError: If fingerprint generation fails
            StorageNotFoundError: If file does not exist in storage

        Example:
            >>> fingerprint = await service.generate_fingerprint(
            ...     asset_id="asset123",
            ...     object_key="uploads/2024/image.jpg",
            ...     file_type="image",
            ...     user_id="user123"
            ... )
            >>> print(fingerprint["fingerprint_id"])  # MongoDB ObjectId
        """
        self.logger.info(
            f"Generating fingerprint for asset_id={asset_id}, "
            f"object_key={object_key}, file_type={file_type}"
        )

        start_time = time.time()
        temp_file_path = None
        fingerprint_id = str(ObjectId())

        try:
            # Normalize file type
            file_type_lower = file_type.lower().strip()

            # Validate file type
            supported_types = {"image", "audio", "video", "text"}
            if file_type_lower not in supported_types:
                raise UnsupportedFileTypeError(
                    f"Unsupported file type: {file_type}. "
                    f"Supported types: {', '.join(supported_types)}"
                )

            # Create temporary file for processing
            suffix = Path(object_key).suffix or ""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file_path = temp_file.name

            self.logger.debug(f"Downloading file to temporary location: {temp_file_path}")

            # Download file from S3 to temporary location
            await self.storage.download_file(object_key, file_path=temp_file_path)

            self.logger.debug("File downloaded successfully, starting fingerprinting")

            # Route to appropriate fingerprinting method
            fingerprint_data: dict[str, Any] = {}

            if file_type_lower == "image":
                fingerprint_data = await self.fingerprint_image(temp_file_path)
            elif file_type_lower == "audio":
                fingerprint_data = await self.fingerprint_audio(temp_file_path)
            elif file_type_lower == "video":
                fingerprint_data = await self.fingerprint_video(temp_file_path)
            elif file_type_lower == "text":
                # For text, read content from file
                with Path(temp_file_path).open(encoding="utf-8", errors="replace") as f:
                    content = f.read()
                fingerprint_data = await self.fingerprint_text(content)

            # Calculate processing duration
            processing_duration = round(time.time() - start_time, 3)

            # Build fingerprint document for MongoDB
            fingerprint_type_enum = FingerprintType(file_type_lower)

            fingerprint_doc = {
                "_id": fingerprint_id,
                "asset_id": asset_id,
                "user_id": user_id,
                "fingerprint_type": fingerprint_type_enum.value,
                "perceptual_hashes": fingerprint_data.get("perceptual_hashes"),
                "spectral_data": fingerprint_data.get("spectral_data"),
                "video_hashes": fingerprint_data.get("video_hashes"),
                "embeddings": fingerprint_data.get("embeddings"),
                "image_metadata": fingerprint_data.get("image_metadata"),
                "audio_metadata": fingerprint_data.get("audio_metadata"),
                "video_metadata": fingerprint_data.get("video_metadata"),
                "text_hash": fingerprint_data.get("text_hash"),
                "text_length": fingerprint_data.get("text_length"),
                "processing_status": ProcessingStatus.COMPLETED.value,
                "processing_duration": processing_duration,
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC),
                # Phase 2 placeholder fields
                "training_detected": None,
                "dataset_matches": None,
                "similarity_scores": None,
                "legal_status": None,
            }

            # Store fingerprint in MongoDB
            db_client = get_db_client()
            fingerprints_collection = db_client.get_fingerprints_collection()

            await fingerprints_collection.insert_one(fingerprint_doc)

            self.logger.info(
                f"Fingerprint stored in MongoDB - fingerprint_id={fingerprint_id}, "
                f"processing_duration={processing_duration}s"
            )

            # Build response
            return {
                "fingerprint_id": fingerprint_id,
                "asset_id": asset_id,
                "user_id": user_id,
                "fingerprint_type": fingerprint_type_enum.value,
                "processing_status": ProcessingStatus.COMPLETED.value,
                "processing_duration": processing_duration,
                **fingerprint_data,
            }

        except UnsupportedFileTypeError:
            # Re-raise without wrapping
            raise
        except Exception as e:
            processing_duration = round(time.time() - start_time, 3)
            error_msg = str(e)

            # Store failed fingerprint record
            try:
                db_client = get_db_client()
                fingerprints_collection = db_client.get_fingerprints_collection()

                error_doc = {
                    "_id": fingerprint_id,
                    "asset_id": asset_id,
                    "user_id": user_id,
                    "fingerprint_type": file_type.lower(),
                    "processing_status": ProcessingStatus.FAILED.value,
                    "error_message": error_msg,
                    "processing_duration": processing_duration,
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                }

                await fingerprints_collection.insert_one(error_doc)
                self.logger.warning(f"Stored failed fingerprint record: {fingerprint_id}")

            except Exception:
                self.logger.exception("Failed to store error record")

            raise FingerprintGenerationError(
                f"Fingerprint generation failed for asset {asset_id}: {error_msg}"
            ) from e

        finally:
            # Cleanup temporary file
            if temp_file_path:
                try:
                    Path(temp_file_path).unlink(missing_ok=True)
                    self.logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as cleanup_error:
                    self.logger.warning(
                        f"Failed to cleanup temporary file {temp_file_path}: {cleanup_error}"
                    )

    # ============================================================================
    # Phase 2 TODO Methods - AI Training Detection Engine
    # ============================================================================

    # TODO Phase 2: Implement AI training detection engine
    async def detect_ai_training_usage(self, fingerprint_id: str) -> dict[str, Any]:
        """
        Detect if an asset's fingerprint appears in known AI training datasets.

        This Phase 2 method will:
        1. Retrieve the fingerprint from MongoDB
        2. Compare embeddings against known AI training dataset fingerprints
        3. Apply similarity thresholds to determine potential matches
        4. Return detection results with confidence scores

        Args:
            fingerprint_id: MongoDB ObjectId of the fingerprint to analyze

        Returns:
            Dictionary containing:
                - detected: Boolean indicating if training usage was detected
                - confidence: Confidence score (0.0-1.0)
                - matched_datasets: List of matched dataset identifiers
                - analysis_timestamp: When analysis was performed

        Raises:
            NotImplementedError: Phase 2 functionality not yet implemented
        """
        self.logger.info(f"AI training detection requested for fingerprint: {fingerprint_id}")
        # TODO Phase 2: Implement AI training detection engine
        raise NotImplementedError("AI training detection will be implemented in Phase 2")

    # TODO Phase 2: Compare embeddings against known AI training datasets
    async def compare_against_training_datasets(
        self, embeddings: list[float]
    ) -> list[dict[str, Any]]:
        """
        Compare embedding vector against known AI training datasets.

        This Phase 2 method will:
        1. Connect to training dataset fingerprint database
        2. Perform efficient similarity search using vector indexes
        3. Return all matches above threshold with similarity scores

        Args:
            embeddings: 1536-dimensional OpenAI embedding vector

        Returns:
            List of dictionaries, each containing:
                - dataset_id: Identifier of matched dataset
                - dataset_name: Human-readable dataset name
                - similarity_score: Cosine similarity (0.0-1.0)
                - match_confidence: Confidence of the match
                - sample_count: Number of similar samples in dataset

        Raises:
            NotImplementedError: Phase 2 functionality not yet implemented
        """
        self.logger.info(f"Dataset comparison requested for embedding ({len(embeddings)} dims)")
        # TODO Phase 2: Compare embeddings against known datasets
        raise NotImplementedError("Dataset comparison will be implemented in Phase 2")

    # TODO Phase 2: Calculate embedding drift scores
    async def calculate_embedding_drift(
        self,
        original_embedding: list[float],
        model_embedding: list[float],
    ) -> float:
        """
        Calculate drift score between original asset embedding and model output embedding.

        This Phase 2 method will:
        1. Compare original creator asset embedding with AI model output embedding
        2. Calculate drift metrics indicating how much the model learned from the original
        3. Return a normalized drift score for compensation calculations

        The drift score helps determine how much influence an original asset
        had on an AI model's outputs, supporting the AI Touch Value™ calculation.

        Args:
            original_embedding: Embedding of original creator asset
            model_embedding: Embedding of AI model output being compared

        Returns:
            float: Drift score from 0.0 (no drift/identical) to 1.0 (maximum drift)

        Raises:
            NotImplementedError: Phase 2 functionality not yet implemented
        """
        self.logger.info(
            f"Embedding drift calculation requested - "
            f"original: {len(original_embedding)} dims, model: {len(model_embedding)} dims"
        )
        # TODO Phase 2: Calculate embedding drift scores
        raise NotImplementedError("Embedding drift analysis will be implemented in Phase 2")

    # TODO Phase 2: Apply similarity-law thresholds for legal determination
    async def apply_similarity_thresholds(self, similarity_score: float) -> dict[str, Any]:
        """
        Apply legal similarity thresholds to determine copyright infringement status.

        This Phase 2 method will:
        1. Apply jurisdiction-specific similarity thresholds
        2. Determine if similarity meets legal thresholds for infringement
        3. Provide legal determination status for creator protection

        Thresholds may vary by:
        - Content type (image, audio, video, text)
        - Jurisdiction (US, EU, etc.)
        - Industry standards and precedents

        Args:
            similarity_score: Similarity score from dataset comparison (0.0-1.0)

        Returns:
            Dictionary containing:
                - threshold_met: Boolean if legal threshold exceeded
                - legal_status: One of "clear", "review_needed", "likely_infringement"
                - confidence: Confidence in determination
                - applicable_thresholds: Thresholds applied by jurisdiction
                - recommended_action: Suggested next steps

        Raises:
            NotImplementedError: Phase 2 functionality not yet implemented
        """
        self.logger.info(f"Similarity threshold analysis requested for score: {similarity_score}")
        # TODO Phase 2: Apply similarity-law thresholds for legal determination
        raise NotImplementedError("Similarity-law thresholds will be implemented in Phase 2")

    # TODO Phase 2: Generate legal-export documentation
    async def generate_legal_export(
        self,
        asset_id: str,
        _detection_results: dict[str, Any],  # Underscore prefix: Unused in Phase 1 stub
    ) -> bytes:
        """
        Generate legal documentation package for court submission.

        This Phase 2 method will:
        1. Compile all fingerprint data for the asset
        2. Include AI training detection results and methodology
        3. Generate court-ready PDF documentation with:
           - Chain of custody for digital evidence
           - Technical methodology explanation
           - Similarity analysis results
           - Expert declarations (templated)
        4. Return PDF document as bytes

        Args:
            asset_id: MongoDB ObjectId of the asset
            detection_results: Results from detect_ai_training_usage

        Returns:
            bytes: PDF document content for legal submission

        Raises:
            NotImplementedError: Phase 2 functionality not yet implemented
        """
        self.logger.info(f"Legal export generation requested for asset: {asset_id}")
        # TODO Phase 2: Generate legal-export documentation
        raise NotImplementedError("Legal export generation will be implemented in Phase 2")

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _extract_video_frame_hashes(
        self,
        cap: cv2.VideoCapture,
        frame_interval: int,
    ) -> tuple[list[str], int]:
        """
        Extract frame hashes from video capture at specified intervals.

        Args:
            cap: OpenCV VideoCapture object
            frame_interval: Number of frames between samples

        Returns:
            Tuple of (list of frame hash strings, number of frames analyzed)
        """
        frame_hashes: list[str] = []
        frame_count = 0
        frames_analyzed = 0

        while True:
            # Limit maximum frames to analyze
            if frames_analyzed >= MAX_VIDEO_FRAMES:
                self.logger.info(f"Reached maximum frame limit ({MAX_VIDEO_FRAMES})")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Only process frames at the sampling interval
            if frame_count % frame_interval == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image for hashing
                pil_image = Image.fromarray(frame_rgb)

                # Resize for consistent hashing
                pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)

                # Generate perceptual hash for frame
                frame_phash = imagehash.phash(pil_image, hash_size=DEFAULT_HASH_SIZE)
                frame_hashes.append(self._normalize_hash(frame_phash))

                frames_analyzed += 1

            frame_count += 1

        return frame_hashes, frames_analyzed

    def _normalize_hash(self, hash_obj: imagehash.ImageHash) -> str:
        """
        Convert imagehash object to normalized hexadecimal string.

        Ensures consistent string representation of perceptual hashes
        for storage and comparison.

        Args:
            hash_obj: ImageHash object from imagehash library

        Returns:
            str: Hexadecimal string representation of hash
        """
        return str(hash_obj)

    def _compute_array_hash(self, array: "np.ndarray[Any, Any]") -> str:
        """
        Compute SHA-256 hash of numpy array for spectral fingerprinting.

        Flattens array, converts to bytes, and computes hash for
        quick comparison of spectral data.

        Args:
            array: Numpy array (e.g., mel-spectrogram or chromagram)

        Returns:
            str: SHA-256 hexadecimal hash of array
        """
        # Flatten and convert to consistent byte representation
        array_bytes = array.flatten().tobytes()
        return hashlib.sha256(array_bytes).hexdigest()

    async def _generate_embedding(self, content: str) -> list[float] | None:
        """
        Generate OpenAI embedding for content if embeddings are configured.

        Uses LangChain OpenAIEmbeddings to generate 1536-dimensional vectors
        for semantic similarity detection.

        Args:
            content: Text content to embed

        Returns:
            List of floats representing embedding vector, or None if:
                - Embeddings not configured
                - Content is empty
                - Generation fails
        """
        if not self.embeddings or not content:
            return None

        try:
            # Use embed_query for single text embedding
            embedding = await self.embeddings.aembed_query(content)
            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            self.logger.warning(f"Failed to generate embedding: {e}")
            return None


# ============================================================================
# Convenience Functions for Standalone Processing
# ============================================================================


async def process_image(file_path: str, openai_api_key: str | None = None) -> dict[str, Any]:
    """
    Standalone function to process an image file and generate fingerprint.

    Creates a minimal FingerprintingService instance for one-off image processing
    without requiring full service initialization.

    Args:
        file_path: Absolute path to the image file
        openai_api_key: Optional OpenAI API key for embedding generation

    Returns:
        Dictionary containing image fingerprint data

    Raises:
        FingerprintGenerationError: If fingerprint generation fails
        FileNotFoundError: If file does not exist
    """
    # Create minimal service without storage dependency
    metadata_service = MetadataService()
    # Create a mock storage service just for the fingerprint_image method
    # which doesn't actually use storage
    service = FingerprintingService.__new__(FingerprintingService)
    service.metadata = metadata_service
    service.logger = logging.getLogger(__name__)
    service.embeddings = None

    if openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
            service.embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        except Exception:
            service.embeddings = None

    return await service.fingerprint_image(file_path)


async def process_audio(file_path: str, openai_api_key: str | None = None) -> dict[str, Any]:
    """
    Standalone function to process an audio file and generate fingerprint.

    Creates a minimal FingerprintingService instance for one-off audio processing
    without requiring full service initialization.

    Args:
        file_path: Absolute path to the audio file
        openai_api_key: Optional OpenAI API key for embedding generation

    Returns:
        Dictionary containing audio fingerprint data

    Raises:
        FingerprintGenerationError: If fingerprint generation fails
        FileNotFoundError: If file does not exist
    """
    metadata_service = MetadataService()
    service = FingerprintingService.__new__(FingerprintingService)
    service.metadata = metadata_service
    service.logger = logging.getLogger(__name__)
    service.embeddings = None

    if openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
            service.embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        except Exception:
            service.embeddings = None

    return await service.fingerprint_audio(file_path)


async def process_video(file_path: str, openai_api_key: str | None = None) -> dict[str, Any]:
    """
    Standalone function to process a video file and generate fingerprint.

    Creates a minimal FingerprintingService instance for one-off video processing
    without requiring full service initialization.

    Args:
        file_path: Absolute path to the video file
        openai_api_key: Optional OpenAI API key for embedding generation

    Returns:
        Dictionary containing video fingerprint data

    Raises:
        FingerprintGenerationError: If fingerprint generation fails
        FileNotFoundError: If file does not exist
    """
    metadata_service = MetadataService()
    service = FingerprintingService.__new__(FingerprintingService)
    service.metadata = metadata_service
    service.logger = logging.getLogger(__name__)
    service.embeddings = None

    if openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
            service.embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        except Exception:
            service.embeddings = None

    return await service.fingerprint_video(file_path)


async def process_text(content: str, openai_api_key: str | None = None) -> dict[str, Any]:
    """
    Standalone function to process text content and generate fingerprint.

    Creates a minimal FingerprintingService instance for one-off text processing
    without requiring full service initialization.

    Args:
        content: Text content to fingerprint
        openai_api_key: Optional OpenAI API key for embedding generation

    Returns:
        Dictionary containing text fingerprint data

    Raises:
        FingerprintGenerationError: If fingerprint generation fails
    """
    service = FingerprintingService.__new__(FingerprintingService)
    service.logger = logging.getLogger(__name__)
    service.embeddings = None

    if openai_api_key:
        try:
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415
            service.embeddings = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        except Exception:
            service.embeddings = None

    return await service.fingerprint_text(content)


# Export all public classes and functions
__all__ = [
    "FingerprintGenerationError",
    "FingerprintingService",
    "FingerprintingServiceError",
    "UnsupportedFileTypeError",
    "process_audio",
    "process_image",
    "process_text",
    "process_video",
]
