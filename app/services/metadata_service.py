"""
Metadata Extraction Service for META-STAMP V3

This service provides comprehensive metadata extraction for all supported file types
including images (EXIF data, dimensions, format), audio files (duration, bitrate,
sample rate), video files (resolution, codec, duration, frame rate), PDFs (author,
pages, creation date), and text files (encoding, size, line count).

This module is part of the asset fingerprinting pipeline and enriches uploaded assets
with contextual metadata for better tracking and analysis.

Author: META-STAMP V3 Platform
License: Proprietary
"""

import logging

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chardet
import cv2
import librosa

from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS
from PyPDF2 import PdfReader


class MetadataService:
    """
    Comprehensive metadata extraction service for all supported file types.

    This service extracts detailed metadata from various file formats to enrich
    asset records with contextual information. Supports:
    - Images: EXIF data, dimensions, format, color mode
    - Audio: Duration, sample rate, bitrate estimation
    - Video: Resolution, frame rate, codec, duration
    - PDF: Author, title, pages, creation/modification dates
    - Text: Encoding detection, size, line count

    All extraction methods are async-compatible for non-blocking operation
    in the FastAPI application context.

    Attributes:
        logger: Logger instance for tracking operations and errors

    Example:
        service = MetadataService()
        metadata = await service.extract_metadata("/path/to/file.jpg", "image")
    """

    def __init__(self) -> None:
        """
        Initialize the MetadataService with logging configuration.

        Sets up a logger instance for tracking metadata extraction operations,
        file processing attempts, errors from PIL/librosa/opencv/PyPDF2,
        missing metadata warnings, and successful extractions.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("MetadataService initialized")

    async def extract_image_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Extract comprehensive metadata from image files.

        Loads the image using PIL and extracts EXIF data, dimensions, format,
        and color mode. EXIF tags are converted from numeric IDs to human-readable
        names using PIL.ExifTags.TAGS mapping.

        Args:
            file_path: Absolute path to the image file

        Returns:
            Dictionary containing:
                - width: Image width in pixels
                - height: Image height in pixels
                - format: Image format (JPEG, PNG, WebP, etc.)
                - mode: Color mode (RGB, RGBA, L, etc.)
                - exif: Dictionary of EXIF data (may be empty if not available)
                - has_exif: Boolean indicating if EXIF data was found
                - file_size: File size in bytes

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid image
        """
        self.logger.info(f"Extracting image metadata from: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"Image file not found: {file_path}")
            raise FileNotFoundError(f"Image file not found: {file_path}")

        if not path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            with Image.open(file_path) as img:
                # Basic image properties
                metadata: dict[str, Any] = {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format,
                    "mode": img.mode,
                    "file_size": path.stat().st_size,
                    "has_exif": False,
                    "exif": {},
                }

                # Extract EXIF data if available
                exif_data = img.getexif()
                if exif_data:
                    metadata["has_exif"] = True
                    exif_dict: dict[str, Any] = {}

                    for tag_id, raw_value in exif_data.items():
                        # Convert tag ID to human-readable name
                        tag_name = TAGS.get(tag_id, str(tag_id))

                        # Handle bytes values by decoding or converting
                        if isinstance(raw_value, bytes):
                            try:
                                tag_value = raw_value.decode("utf-8", errors="replace")
                            except Exception:
                                tag_value = str(raw_value)
                        else:
                            tag_value = raw_value

                        # Store the tag with its value
                        exif_dict[tag_name] = tag_value

                    metadata["exif"] = exif_dict
                    self.logger.debug(f"Extracted {len(exif_dict)} EXIF tags")
                else:
                    self.logger.debug("No EXIF data found in image")

                self.logger.info(
                    f"Image metadata extracted: {metadata['width']}x{metadata['height']} "
                    f"{metadata['format']} ({metadata['mode']})"
                )
                return metadata

        except UnidentifiedImageError as e:
            self.logger.exception("Cannot identify image file: %s", file_path)
            raise ValueError(f"Cannot identify image file: {file_path}") from e
        except Exception:
            self.logger.exception("Error extracting image metadata from %s", file_path)
            raise

    async def extract_audio_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Extract comprehensive metadata from audio files.

        Loads the audio file using librosa and extracts duration, sample rate,
        and estimates bitrate based on file size and duration. Supports common
        audio formats like MP3, WAV, AAC, FLAC, etc.

        Args:
            file_path: Absolute path to the audio file

        Returns:
            Dictionary containing:
                - duration: Duration in seconds (float)
                - duration_formatted: Human-readable duration (HH:MM:SS)
                - sample_rate: Sample rate in Hz
                - estimated_bitrate_kbps: Estimated bitrate in kbps
                - file_size: File size in bytes
                - channels: Number of audio channels (estimated)

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid audio file
        """
        self.logger.info(f"Extracting audio metadata from: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"Audio file not found: {file_path}")
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            # Load audio file with librosa
            # sr=None preserves the original sample rate
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False)

            # Get duration using librosa
            duration = librosa.get_duration(y=audio_data, sr=sample_rate)

            # Get file size for bitrate estimation
            file_size = path.stat().st_size

            # Estimate bitrate (bits per second, then convert to kbps)
            # bitrate = (file_size_bytes * 8) / duration_seconds / 1000
            estimated_bitrate_kbps = 0.0
            if duration > 0:
                estimated_bitrate_kbps = round((file_size * 8) / duration / 1000, 2)

            # Determine number of channels
            channels = 1 if audio_data.ndim == 1 else audio_data.shape[0]

            # Format duration as HH:MM:SS
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            metadata: dict[str, Any] = {
                "duration": round(duration, 3),
                "duration_formatted": duration_formatted,
                "sample_rate": sample_rate,
                "estimated_bitrate_kbps": estimated_bitrate_kbps,
                "file_size": file_size,
                "channels": channels,
            }

            self.logger.info(
                f"Audio metadata extracted: {duration_formatted} @ "
                f"{sample_rate}Hz, ~{estimated_bitrate_kbps}kbps"
            )
            return metadata

        except Exception as e:
            self.logger.exception("Error extracting audio metadata from %s", file_path)
            raise ValueError(f"Error extracting audio metadata: {e}") from e

    async def extract_video_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Extract comprehensive metadata from video files.

        Opens the video file using OpenCV VideoCapture and extracts resolution,
        frame rate, codec information, frame count, and calculated duration.
        Supports common video formats like MP4, MOV, AVI, etc.

        Args:
            file_path: Absolute path to the video file

        Returns:
            Dictionary containing:
                - width: Video width in pixels
                - height: Video height in pixels
                - resolution: Formatted resolution string (e.g., "1920x1080")
                - frame_rate: Frames per second (float)
                - frame_count: Total number of frames
                - duration: Duration in seconds (calculated from frames/fps)
                - duration_formatted: Human-readable duration (HH:MM:SS)
                - codec: FourCC codec identifier
                - codec_name: Human-readable codec name if available
                - file_size: File size in bytes

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid video or cannot be opened
        """
        self.logger.info(f"Extracting video metadata from: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"Video file not found: {file_path}")
            raise FileNotFoundError(f"Video file not found: {file_path}")

        if not path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        cap = None
        try:
            # Open video file with OpenCV
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                self.logger.error(f"Cannot open video file: {file_path}")
                raise ValueError(f"Cannot open video file: {file_path}")

            # Extract video properties using CAP_PROP constants
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Extract codec information using FourCC
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            # Convert FourCC integer to string
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

            # Calculate duration from frame count and fps
            duration = 0.0
            if frame_rate > 0:
                duration = frame_count / frame_rate

            # Format duration as HH:MM:SS
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Get file size
            file_size = path.stat().st_size

            # Map common FourCC codes to readable names
            codec_names: dict[str, str] = {
                "avc1": "H.264/AVC",
                "h264": "H.264",
                "hevc": "H.265/HEVC",
                "hvc1": "H.265/HEVC",
                "mp4v": "MPEG-4",
                "xvid": "Xvid",
                "divx": "DivX",
                "vp8 ": "VP8",
                "vp9 ": "VP9",
                "av01": "AV1",
                "mjpg": "Motion JPEG",
            }
            codec_name = codec_names.get(codec.lower().strip(), codec)

            metadata: dict[str, Any] = {
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "frame_rate": round(frame_rate, 2),
                "frame_count": frame_count,
                "duration": round(duration, 3),
                "duration_formatted": duration_formatted,
                "codec": codec,
                "codec_name": codec_name,
                "file_size": file_size,
            }

            self.logger.info(
                f"Video metadata extracted: {width}x{height} @ "
                f"{frame_rate:.2f}fps, {duration_formatted}, {codec_name}"
            )
            return metadata

        except cv2.error as e:
            self.logger.exception("OpenCV error extracting video metadata")
            raise ValueError(f"Error extracting video metadata: {e}") from e
        except Exception:
            self.logger.exception("Error extracting video metadata from %s", file_path)
            raise
        finally:
            # Always release the video capture
            if cap is not None:
                cap.release()

    async def extract_pdf_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Extract comprehensive metadata from PDF files.

        Opens the PDF file using PyPDF2 PdfReader and extracts document
        information including author, title, subject, page count, and
        creation/modification dates.

        Args:
            file_path: Absolute path to the PDF file

        Returns:
            Dictionary containing:
                - page_count: Total number of pages
                - author: Document author (None if not set)
                - title: Document title (None if not set)
                - subject: Document subject (None if not set)
                - creator: Application that created the PDF (None if not set)
                - producer: PDF producer/converter (None if not set)
                - creation_date: Document creation date (None if not set)
                - modification_date: Last modification date (None if not set)
                - is_encrypted: Boolean indicating if PDF is encrypted
                - file_size: File size in bytes

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is not a valid PDF
        """
        self.logger.info(f"Extracting PDF metadata from: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"PDF file not found: {file_path}")
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            with path.open("rb") as pdf_file:
                reader = PdfReader(pdf_file)

                # Get page count
                page_count = len(reader.pages)

                # Get document information dictionary
                doc_info = reader.metadata

                # Extract metadata fields with None defaults
                author: str | None = None
                title: str | None = None
                subject: str | None = None
                creator: str | None = None
                producer: str | None = None
                creation_date: str | None = None
                modification_date: str | None = None

                if doc_info:
                    author = doc_info.author
                    title = doc_info.title
                    subject = doc_info.subject
                    creator = doc_info.creator
                    producer = doc_info.producer

                    # Handle creation date
                    if doc_info.creation_date:
                        try:
                            creation_date = doc_info.creation_date.isoformat()
                        except (AttributeError, TypeError):
                            creation_date = str(doc_info.creation_date)

                    # Handle modification date
                    if doc_info.modification_date:
                        try:
                            modification_date = doc_info.modification_date.isoformat()
                        except (AttributeError, TypeError):
                            modification_date = str(doc_info.modification_date)

                # Check if PDF is encrypted
                is_encrypted = reader.is_encrypted

                # Get file size
                file_size = path.stat().st_size

                metadata: dict[str, Any] = {
                    "page_count": page_count,
                    "author": author,
                    "title": title,
                    "subject": subject,
                    "creator": creator,
                    "producer": producer,
                    "creation_date": creation_date,
                    "modification_date": modification_date,
                    "is_encrypted": is_encrypted,
                    "file_size": file_size,
                }

                self.logger.info(
                    f"PDF metadata extracted: {page_count} pages, "
                    f"author={author}, encrypted={is_encrypted}"
                )
                return metadata

        except Exception as e:
            self.logger.exception("Error extracting PDF metadata from %s", file_path)
            raise ValueError(f"Error extracting PDF metadata: {e}") from e

    async def extract_text_metadata(self, file_path: str) -> dict[str, Any]:
        """
        Extract comprehensive metadata from text files.

        Analyzes the text file to detect encoding using chardet, counts lines
        and characters, and determines file size. Supports plain text files,
        Markdown files, and other text-based formats.

        Args:
            file_path: Absolute path to the text file

        Returns:
            Dictionary containing:
                - encoding: Detected file encoding (UTF-8, ASCII, ISO-8859-1, etc.)
                - encoding_confidence: Confidence level of encoding detection (0-1)
                - file_size: File size in bytes
                - line_count: Number of lines in the file
                - character_count: Total number of characters
                - word_count: Approximate word count
                - is_empty: Boolean indicating if file is empty

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file cannot be read
        """
        self.logger.info(f"Extracting text metadata from: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"Text file not found: {file_path}")
            raise FileNotFoundError(f"Text file not found: {file_path}")

        if not path.is_file():
            self.logger.error(f"Path is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        try:
            # Get file size
            file_size = path.stat().st_size

            # Read raw bytes for encoding detection
            with path.open("rb") as f:
                raw_content = f.read()

            # Detect encoding using chardet
            detection_result = chardet.detect(raw_content)
            encoding = detection_result.get("encoding", "utf-8") or "utf-8"
            encoding_confidence = detection_result.get("confidence", 0.0) or 0.0

            # Decode content for analysis
            try:
                content = raw_content.decode(encoding, errors="replace")
            except (UnicodeDecodeError, LookupError):
                # Fallback to utf-8 with replacement
                content = raw_content.decode("utf-8", errors="replace")
                encoding = "utf-8 (fallback)"

            # Count lines
            lines = content.splitlines()
            line_count = len(lines)

            # Count characters (excluding line endings for consistency)
            character_count = len(content)

            # Count words (split on whitespace)
            word_count = len(content.split())

            # Check if file is empty
            is_empty = file_size == 0 or character_count == 0

            metadata: dict[str, Any] = {
                "encoding": encoding,
                "encoding_confidence": round(encoding_confidence, 3),
                "file_size": file_size,
                "line_count": line_count,
                "character_count": character_count,
                "word_count": word_count,
                "is_empty": is_empty,
            }

            self.logger.info(
                f"Text metadata extracted: {encoding} encoding, "
                f"{line_count} lines, {word_count} words"
            )
            return metadata

        except Exception as e:
            self.logger.exception("Error extracting text metadata from %s", file_path)
            raise ValueError(f"Error extracting text metadata: {e}") from e

    async def extract_metadata(self, file_path: str, file_type: str) -> dict[str, Any]:
        """
        Universal metadata extractor that routes to the appropriate method.

        Determines the appropriate extraction method based on file_type and
        extracts metadata. Also adds common metadata like file size, modification
        time, and file extension regardless of type.

        Supported file types:
            - "image": Routes to extract_image_metadata (JPEG, PNG, WebP)
            - "audio": Routes to extract_audio_metadata (MP3, WAV, AAC)
            - "video": Routes to extract_video_metadata (MP4, MOV, AVI)
            - "pdf": Routes to extract_pdf_metadata (PDF)
            - "text": Routes to extract_text_metadata (TXT, MD)

        Args:
            file_path: Absolute path to the file
            file_type: Type of file ("image", "audio", "video", "pdf", "text")

        Returns:
            Dictionary containing:
                - type_specific metadata (depends on file_type)
                - file_path: Original file path
                - file_type: File type identifier
                - file_extension: File extension (e.g., ".jpg")
                - modification_time: File modification timestamp (ISO format)
                - extraction_timestamp: When metadata was extracted (ISO format)

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If file_type is unsupported or file is invalid
        """
        self.logger.info(f"Extracting metadata for file_type={file_type} from: {file_path}")

        path = Path(file_path)
        if not path.exists():
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Normalize file type to lowercase
        file_type_lower = file_type.lower().strip()

        # Route to appropriate extractor based on file_type
        type_specific_metadata: dict[str, Any] = {}

        try:
            if file_type_lower == "image":
                type_specific_metadata = await self.extract_image_metadata(file_path)
            elif file_type_lower == "audio":
                type_specific_metadata = await self.extract_audio_metadata(file_path)
            elif file_type_lower == "video":
                type_specific_metadata = await self.extract_video_metadata(file_path)
            elif file_type_lower == "pdf":
                type_specific_metadata = await self.extract_pdf_metadata(file_path)
            elif file_type_lower == "text":
                type_specific_metadata = await self.extract_text_metadata(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_type}")
                raise ValueError(
                    f"Unsupported file type: {file_type}. "
                    f"Supported types: image, audio, video, pdf, text"
                )
        except (FileNotFoundError, ValueError):
            # Re-raise these exceptions as-is
            raise
        except Exception as e:
            self.logger.exception("Error during type-specific metadata extraction")
            # Store error info but continue with common metadata
            type_specific_metadata = {"extraction_error": str(e), "extraction_successful": False}

        # Add common metadata
        stat_info = path.stat()
        modification_time = datetime.fromtimestamp(stat_info.st_mtime, tz=UTC).isoformat()

        common_metadata: dict[str, Any] = {
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "file_type": file_type_lower,
            "file_extension": path.suffix.lower(),
            "file_size": stat_info.st_size,
            "modification_time": modification_time,
            "extraction_timestamp": datetime.now(tz=UTC).isoformat(),
        }

        # Merge type-specific and common metadata
        # Type-specific takes precedence for overlapping keys (like file_size)
        metadata = {**common_metadata, **type_specific_metadata}

        # Ensure extraction was successful if no error was recorded
        if "extraction_successful" not in metadata:
            metadata["extraction_successful"] = True

        self.logger.info(
            f"Metadata extraction complete for {file_type_lower}: "
            f"{path.name} ({metadata.get('file_size', 0)} bytes)"
        )

        return metadata


# Export the service class
__all__ = ["MetadataService"]
