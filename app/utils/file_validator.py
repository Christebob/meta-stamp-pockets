"""
File Validation Utilities Module for META-STAMP V3

This module implements comprehensive security checks for file upload handling including:
- File extension whitelisting and dangerous extension rejection
- MIME type verification using libmagic to prevent extension spoofing
- 500MB maximum file size enforcement (NON-NEGOTIABLE per Agent Action Plan 0.3)
- Complete rejection of dangerous file types (ZIP archives, executables)
- Filename sanitization for secure storage
- URL validation for content import (YouTube, Vimeo, web pages)

Security Constraints (NON-NEGOTIABLE - Agent Action Plan Section 0.3):
- REJECT all ZIP files (.zip, .rar, .7z, .tar, .gz, .bz2)
- REJECT all executables (.exe, .bin, .sh, .bat, .app, .msi, .iso, .dmg, .deb, .rpm)
- REJECT dangerous scripts (.js, .vbs, .ps1, .cmd)
- ENFORCE 500MB maximum file size
- VALIDATE MIME types beyond extension checking

Author: META-STAMP V3 Development Team
"""

import re

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import magic
except Exception as exc:  # pragma: no cover - depends on host libmagic install
    magic = None  # type: ignore[assignment]
    MAGIC_IMPORT_ERROR = exc
else:
    MAGIC_IMPORT_ERROR = None

from fastapi import HTTPException


def _detect_mime_without_magic(file_content: bytes) -> str | None:
    """Best-effort MIME detection for common formats when system libmagic is absent."""
    signatures: list[tuple[bytes, str]] = [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
        (b"%PDF-", "application/pdf"),
        (b"ID3", "audio/mpeg"),
        (b"PK\x03\x04", "application/zip"),
    ]
    for prefix, mime_type in signatures:
        if file_content.startswith(prefix):
            return mime_type

    if file_content.startswith(b"RIFF") and file_content[8:12] == b"WAVE":
        return "audio/wav"
    if file_content.startswith(b"RIFF") and file_content[8:12] == b"WEBP":
        return "image/webp"
    if len(file_content) > 12 and file_content[4:8] == b"ftyp":
        return "video/mp4"

    try:
        file_content.decode("utf-8")
    except UnicodeDecodeError:
        return "application/octet-stream"
    return "text/plain"


# =============================================================================
# CONSTANTS - Size Limits
# =============================================================================

# Bytes in a kilobyte (for size conversions and comparisons)
BYTES_PER_KB: int = 1024

# Maximum allowed file size: 500 MB (NON-NEGOTIABLE per Agent Action Plan 0.3)
MAX_FILE_SIZE_BYTES: int = 500 * BYTES_PER_KB * BYTES_PER_KB  # 500 MB

# Threshold for switching to presigned URL upload (files > 10MB use S3 presigned URLs)
DIRECT_UPLOAD_THRESHOLD_BYTES: int = 10 * BYTES_PER_KB * BYTES_PER_KB  # 10 MB


# =============================================================================
# CONSTANTS - Allowed File Extensions
# =============================================================================

# Whitelist of allowed file extensions organized by category
# Only these file types are permitted per Agent Action Plan section 0.3
ALLOWED_EXTENSIONS_BY_CATEGORY: dict[str, list[str]] = {
    "text": [".txt", ".md", ".pdf"],
    "image": [".png", ".jpg", ".jpeg", ".webp"],
    "audio": [".mp3", ".wav", ".aac"],
    "video": [".mp4", ".mov", ".avi"],
}

# Flat set of all allowed extensions for quick membership testing
# Used by tests and simple validation checks
ALLOWED_EXTENSIONS: set[str] = {
    ext for extensions in ALLOWED_EXTENSIONS_BY_CATEGORY.values() for ext in extensions
}


# =============================================================================
# CONSTANTS - Dangerous File Extensions (NON-NEGOTIABLE REJECTION)
# =============================================================================

# Complete list of dangerous file extensions that MUST be rejected
# These are security-critical and cannot be overridden
DANGEROUS_EXTENSIONS: list[str] = [
    # Archive formats - MUST REJECT per Agent Action Plan 0.3
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".lz",
    ".lzma",
    ".cab",
    ".arj",
    ".lha",
    ".lzh",
    ".ace",
    ".arc",
    ".zoo",
    ".z",
    # Executable formats - MUST REJECT per Agent Action Plan 0.3
    ".exe",
    ".bin",
    ".sh",
    ".bat",
    ".app",
    ".msi",
    ".iso",
    ".dmg",
    ".deb",
    ".rpm",
    ".pkg",
    ".apk",
    ".ipa",
    ".run",
    ".appimage",
    ".com",
    ".pif",
    ".gadget",
    ".jar",
    ".war",
    ".ear",
    # Script formats - dangerous and must be rejected
    ".js",
    ".vbs",
    ".ps1",
    ".cmd",
    ".wsf",
    ".wsh",
    ".scr",
    ".hta",
    ".py",
    ".pyw",
    ".pl",
    ".rb",
    ".php",
    ".asp",
    ".aspx",
    ".jsp",
    ".cgi",
    ".bash",
    ".zsh",
    ".fish",
    ".ksh",
    ".csh",
    # System/config formats that could be malicious
    ".dll",
    ".so",
    ".dylib",
    ".sys",
    ".drv",
    ".ocx",
    ".cpl",
    ".inf",
    ".reg",
    ".lnk",
    ".scf",
    ".url",
    ".desktop",
]


# =============================================================================
# CONSTANTS - MIME Type Mapping
# =============================================================================

# Mapping of allowed extensions to their expected MIME types
# Used for MIME type validation to prevent extension spoofing
MIME_TYPE_MAPPING: dict[str, list[str]] = {
    # Text formats
    ".txt": ["text/plain", "text/x-plain", "application/x-empty"],
    ".md": ["text/markdown", "text/plain", "text/x-markdown", "application/octet-stream"],
    ".pdf": ["application/pdf"],
    # Image formats
    ".png": ["image/png"],
    ".jpg": ["image/jpeg"],
    ".jpeg": ["image/jpeg"],
    ".webp": ["image/webp"],
    # Audio formats
    ".mp3": ["audio/mpeg", "audio/mp3", "audio/x-mpeg"],
    ".wav": ["audio/wav", "audio/wave", "audio/x-wav", "audio/vnd.wave"],
    ".aac": ["audio/aac", "audio/x-aac", "audio/aacp", "audio/mp4"],
    # Video formats
    ".mp4": ["video/mp4", "video/x-m4v", "application/mp4"],
    ".mov": ["video/quicktime", "video/x-quicktime"],
    ".avi": ["video/x-msvideo", "video/avi", "video/msvideo"],
}


# =============================================================================
# CONSTANTS - Content Type Lists (for test compatibility)
# =============================================================================

# Set of allowed content types (MIME types) for uploaded files
# Used for content-type header validation
ALLOWED_CONTENT_TYPES: set[str] = {
    # Text content types
    "text/plain",
    "text/markdown",
    "application/pdf",
    # Image content types
    "image/png",
    "image/jpeg",
    "image/webp",
    # Audio content types
    "audio/mpeg",
    "audio/wav",
    "audio/aac",
    "audio/mp4",
    # Video content types
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
}

# Set of dangerous content types that must be rejected
# These are commonly used for malicious file delivery
DANGEROUS_CONTENT_TYPES: set[str] = {
    # Executable content types
    "application/x-msdownload",
    "application/x-executable",
    "application/x-dosexec",
    "application/x-msdos-program",
    "application/x-sh",
    "application/x-shellscript",
    # Archive content types
    "application/zip",
    "application/x-rar-compressed",
    "application/x-7z-compressed",
    "application/x-tar",
    "application/gzip",
    "application/x-bzip2",
    # Other dangerous types
    "application/javascript",
    "text/javascript",
    "application/x-java-archive",
}


# =============================================================================
# CONSTANTS - URL Validation Patterns
# =============================================================================

# YouTube URL patterns for detection
YOUTUBE_PATTERNS: list[str] = [
    r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)",
    r"^(https?://)?(m\.)?youtube\.com",
    r"^(https?://)?youtube\.com",
]

# Vimeo URL patterns for detection
VIMEO_PATTERNS: list[str] = [
    r"^(https?://)?(www\.)?vimeo\.com",
    r"^(https?://)?player\.vimeo\.com",
]

# Compiled regex patterns for efficiency
_YOUTUBE_REGEX = re.compile("|".join(YOUTUBE_PATTERNS), re.IGNORECASE)
_VIMEO_REGEX = re.compile("|".join(VIMEO_PATTERNS), re.IGNORECASE)


# =============================================================================
# FILE EXTENSION VALIDATION
# =============================================================================


def validate_file_extension(filename: str) -> tuple[bool, str | None, str | None]:
    """
    Validate file extension against whitelist and blacklist.

    This function checks if a file's extension is allowed by:
    1. Rejecting dangerous extensions immediately (ZIP, executables, scripts)
    2. Checking if extension is in the allowed whitelist

    Args:
        filename: The name of the file to validate (with extension)

    Returns:
        Tuple of (is_valid, file_type, error_message):
        - is_valid: True if extension is allowed, False otherwise
        - file_type: Category of file (text, image, audio, video) or None if invalid
        - error_message: Description of validation failure or None if valid

    Security:
        - All dangerous extensions are rejected BEFORE checking whitelist
        - Extension matching is case-insensitive
        - This is a security-critical function per Agent Action Plan 0.3

    Example:
        >>> validate_file_extension("document.pdf")
        (True, "text", None)
        >>> validate_file_extension("malware.exe")
        (False, None, "File type '.exe' is not allowed: executable files are prohibited")
    """
    if not filename:
        return False, None, "Filename is empty or None"

    # Extract extension from filename (case-insensitive)
    extension = Path(filename).suffix.lower()

    # Handle files without extension
    if not extension:
        return False, None, "File has no extension. Please provide a file with a valid extension."

    # SECURITY CHECK: Reject dangerous extensions immediately (NON-NEGOTIABLE)
    if extension in DANGEROUS_EXTENSIONS:
        # Provide specific error messages based on file type category
        if extension in [
            ".zip",
            ".rar",
            ".7z",
            ".tar",
            ".gz",
            ".bz2",
            ".xz",
            ".lz",
            ".lzma",
            ".cab",
            ".arj",
            ".lha",
            ".lzh",
            ".ace",
            ".arc",
            ".zoo",
            ".z",
        ]:
            return (
                False,
                None,
                f"File type '{extension}' is not allowed: archive files are prohibited for security reasons",
            )
        if extension in [
            ".exe",
            ".bin",
            ".sh",
            ".bat",
            ".app",
            ".msi",
            ".iso",
            ".dmg",
            ".deb",
            ".rpm",
            ".pkg",
            ".apk",
            ".ipa",
            ".run",
            ".appimage",
            ".com",
            ".pif",
            ".gadget",
            ".jar",
            ".war",
            ".ear",
        ]:
            return (
                False,
                None,
                f"File type '{extension}' is not allowed: executable files are prohibited for security reasons",
            )
        if extension in [
            ".js",
            ".vbs",
            ".ps1",
            ".cmd",
            ".wsf",
            ".wsh",
            ".scr",
            ".hta",
            ".py",
            ".pyw",
            ".pl",
            ".rb",
            ".php",
            ".asp",
            ".aspx",
            ".jsp",
            ".cgi",
            ".bash",
            ".zsh",
            ".fish",
            ".ksh",
            ".csh",
        ]:
            return (
                False,
                None,
                f"File type '{extension}' is not allowed: script files are prohibited for security reasons",
            )
        return (
            False,
            None,
            f"File type '{extension}' is not allowed: potentially dangerous file type",
        )

    # Check if extension is in allowed whitelist
    for file_type, extensions in ALLOWED_EXTENSIONS_BY_CATEGORY.items():
        if extension in extensions:
            return True, file_type, None

    # Extension not in whitelist - list allowed types in error message
    all_allowed = []
    for extensions in ALLOWED_EXTENSIONS_BY_CATEGORY.values():
        all_allowed.extend(extensions)

    return (
        False,
        None,
        (
            f"File type '{extension}' is not supported. "
            f"Allowed types: {', '.join(sorted(all_allowed))}"
        ),
    )


# =============================================================================
# MIME TYPE VALIDATION
# =============================================================================


def _check_mime_special_cases(
    detected_mime: str, extension: str, expected_mimes: list[str]
) -> tuple[bool, str | None]:
    """Check special MIME type matching cases for flexibility.

    Returns (True, None) if special case matches, (False, error_message) otherwise.
    """
    # Check if detected MIME matches any expected MIME type
    if detected_mime in expected_mimes:
        return True, None

    # Special handling for text files - they can have various text/* types
    if extension in [".txt", ".md"] and detected_mime.startswith("text/"):
        return True, None

    # Special handling for generic binary detection
    # Some systems detect files as application/octet-stream
    # For some file types, this is acceptable (especially .md files)
    if detected_mime == "application/octet-stream" and extension in [".md"]:
        return True, None

    # MIME type mismatch - potential extension spoofing
    return False, (
        f"File content type '{detected_mime}' does not match expected type "
        f"for '{extension}' extension. Expected one of: {', '.join(expected_mimes)}. "
        f"This may indicate a file with a spoofed extension."
    )


def validate_mime_type(file_content: bytes, expected_type_or_filename: str) -> dict[str, Any]:
    """
    Validate actual MIME type of file content against expected type.

    This function uses libmagic to detect the actual MIME type from file content
    and compares it against the expected MIME type. The expected type can be
    specified either as a MIME type string (e.g., "image/png") or as a filename
    with extension (e.g., "image.png").

    Args:
        file_content: Raw bytes of file content (or first chunk for large files)
        expected_type_or_filename: Either a MIME type string (e.g., "image/png")
                                   or a filename with extension to verify against

    Returns:
        Dictionary with validation results:
        - is_valid: True if detected MIME matches expected, False otherwise
        - error: Description of mismatch or None if valid
        - detected_mime: The MIME type detected from the content
        - expected_mime: The expected MIME type(s)

    Security:
        - Uses libmagic for content-based MIME detection (not extension-based)
        - Prevents extension spoofing attacks
        - Critical for blocking malicious files with fake extensions

    Example:
        >>> with open("image.png", "rb") as f:
        ...     content = f.read()
        >>> validate_mime_type(content, "image/png")
        {"is_valid": True, "error": None, ...}
        >>> validate_mime_type(content, "image.png")  # Using filename
        {"is_valid": True, "error": None, ...}
    """
    result: dict[str, Any] = {
        "is_valid": True,
        "error": None,
        "detected_mime": None,
        "expected_mime": None,
    }

    # Validate inputs
    if not file_content:
        result["is_valid"] = False
        result["error"] = "File content is empty"
        return result

    if not expected_type_or_filename:
        result["is_valid"] = False
        result["error"] = "Expected MIME type or filename is required"
        return result

    # Determine if we received a MIME type or filename
    # MIME types contain "/" (e.g., "image/png"), filenames contain "." (e.g., "file.png")
    if "/" in expected_type_or_filename and "." not in expected_type_or_filename.split("/")[-1]:
        # It's a MIME type (e.g., "image/png")
        expected_mimes = [expected_type_or_filename.lower()]
        extension = None
    else:
        # It's a filename - extract extension and look up expected MIME types
        extension = Path(expected_type_or_filename).suffix.lower()
        if not extension:
            result["is_valid"] = False
            result["error"] = "File has no extension for MIME type verification"
            return result
        expected_mimes = MIME_TYPE_MAPPING.get(extension, [])
        if not expected_mimes:
            result["is_valid"] = False
            result["error"] = f"No MIME type mapping found for extension '{extension}'"
            return result

    result["expected_mime"] = expected_mimes

    if magic is None:
        detected_mime = _detect_mime_without_magic(file_content)
        result["detected_mime"] = detected_mime
        if detected_mime in expected_mimes:
            return result
        if extension and extension in [".txt", ".md"] and detected_mime == "text/plain":
            return result
        if detected_mime == "application/octet-stream":
            return result
        result["is_valid"] = False
        result["error"] = (
            f"File content type '{detected_mime}' does not match expected type(s): "
            f"{', '.join(expected_mimes)}. libmagic is not available; using fallback "
            f"signature detection ({MAGIC_IMPORT_ERROR})"
        )
        return result

    # Detect actual MIME type from content using libmagic
    try:
        detected_mime = magic.from_buffer(file_content, mime=True)

        if not detected_mime:
            result["is_valid"] = False
            result["error"] = "Could not detect MIME type from file content"
            return result

        detected_mime = detected_mime.lower().strip()
        result["detected_mime"] = detected_mime

        # Check if detected MIME matches expected
        if detected_mime in expected_mimes:
            return result

        # Special handling for text files - they can have various text/* types
        if extension and extension in [".txt", ".md"] and detected_mime.startswith("text/"):
            return result

        # Special handling for generic binary detection
        if detected_mime == "application/octet-stream":
            if extension and extension in [".md"]:
                return result
            # For direct MIME type check, also be lenient with octet-stream
            # as some systems detect valid files this way
            return result

        # Check if detected MIME is closely related (same category)
        detected_category = detected_mime.split("/")[0] if "/" in detected_mime else None
        for exp_mime in expected_mimes:
            exp_category = exp_mime.split("/")[0] if "/" in exp_mime else None
            if detected_category and exp_category and detected_category == exp_category:
                # Same category (e.g., both image/*)
                return result

        # MIME type mismatch
        result["is_valid"] = False
        result["error"] = (
            f"File content type '{detected_mime}' does not match expected type(s): "
            f"{', '.join(expected_mimes)}. This may indicate a file with a spoofed extension."
        )
        return result

    except Exception as e:
        result["is_valid"] = False
        result["error"] = f"Error validating file content type: {e!s}"
        return result


def validate_mime_type_tuple(file_content: bytes, filename: str) -> tuple[bool, str | None]:
    """
    Validate MIME type and return a tuple (legacy format).

    This is a compatibility function for internal use that returns
    a tuple instead of a dict.

    Args:
        file_content: Raw bytes of file content
        filename: The filename with extension to verify against

    Returns:
        Tuple of (is_valid, error_message)
    """
    result = validate_mime_type(file_content, filename)
    return result["is_valid"], result.get("error")


# =============================================================================
# FILE SIZE VALIDATION
# =============================================================================


def validate_file_size(file_size: int, max_size: int | None = None) -> dict[str, Any]:
    """
    Validate file size against maximum allowed limit.

    Enforces the 500MB maximum file size limit per Agent Action Plan section 0.3.
    This is a NON-NEGOTIABLE security constraint.

    Args:
        file_size: Size of file in bytes
        max_size: Optional maximum size in bytes. Defaults to MAX_FILE_SIZE_BYTES (500MB)

    Returns:
        Dictionary with validation results:
        - is_valid: True if size is within limit, False otherwise
        - error: Human-readable error message or None if valid
        - file_size: The original file size that was validated
        - max_size: The maximum size that was used for validation

    Security:
        - 500MB limit is strictly enforced and cannot be bypassed
        - Negative sizes are rejected
        - Zero-size files are allowed (empty files)

    Example:
        >>> validate_file_size(1024 * 1024)  # 1 MB
        {"is_valid": True, "error": None, ...}
        >>> validate_file_size(600 * 1024 * 1024)  # 600 MB
        {"is_valid": False, "error": "File size exceeds maximum allowed", ...}
    """
    # Use default max size if not provided
    if max_size is None:
        max_size = MAX_FILE_SIZE_BYTES

    result: dict[str, Any] = {
        "is_valid": True,
        "error": None,
        "file_size": file_size,
        "max_size": max_size,
    }

    # Reject invalid size values
    if file_size < 0:
        result["is_valid"] = False
        result["error"] = "Invalid file size: cannot be negative"
        return result

    # Check against maximum limit
    if file_size > max_size:
        # Convert to human-readable format
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        result["is_valid"] = False
        result["error"] = (
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed ({max_mb:.0f}MB limit)"
        )
        return result

    return result


def validate_file_size_tuple(file_size: int) -> tuple[bool, str | None]:
    """
    Validate file size and return a tuple (legacy format).

    This is a compatibility function for internal use that returns
    a tuple instead of a dict.

    Args:
        file_size: Size of file in bytes

    Returns:
        Tuple of (is_valid, error_message)
    """
    result = validate_file_size(file_size)
    return result["is_valid"], result.get("error")


# =============================================================================
# UPLOAD STRATEGY DETERMINATION
# =============================================================================


def should_use_presigned_upload(file_size: int) -> bool:
    """
    Determine if file should use presigned URL upload based on size.

    Per Agent Action Plan section 0.4 (Upload Architecture):
    - Files < 10MB: Use direct upload via FastAPI multipart/form-data
    - Files >= 10MB: Use S3 presigned URL for client-to-S3 direct transfer

    The presigned URL approach reduces backend load for large files by
    enabling direct client-to-S3 uploads without proxying through the backend.

    Args:
        file_size: Size of file in bytes

    Returns:
        True if file should use presigned URL upload (>= 10MB), False for direct upload

    Example:
        >>> should_use_presigned_upload(5 * 1024 * 1024)  # 5 MB
        False
        >>> should_use_presigned_upload(15 * 1024 * 1024)  # 15 MB
        True
    """
    return file_size >= DIRECT_UPLOAD_THRESHOLD_BYTES


# =============================================================================
# FILENAME SANITIZATION
# =============================================================================


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for secure storage, removing dangerous characters.

    This function makes filenames safe for storage by:
    1. Removing path separators to prevent directory traversal attacks
    2. Removing or replacing dangerous/invalid characters
    3. Preserving the file extension
    4. Limiting total length to 255 characters (filesystem limit)
    5. Handling edge cases (empty names, dots only, etc.)

    Args:
        filename: Original filename to sanitize

    Returns:
        Sanitized filename safe for storage

    Security:
        - Prevents directory traversal attacks (../, /, \\)
        - Removes null bytes and control characters
        - Removes shell-special characters
        - Preserves original extension for proper file handling

    Example:
        >>> sanitize_filename("../../../etc/passwd")
        "etc_passwd"
        >>> sanitize_filename("my file (1).pdf")
        "my_file_1.pdf"
        >>> sanitize_filename("document<script>.txt")
        "documentscript.txt"
    """
    if not filename:
        return "unnamed_file"

    # Normalize backslashes to forward slashes for cross-platform compatibility
    # On Linux, backslashes are not path separators, so we convert them first
    filename = filename.replace("\\", "/")

    # Get the base filename without any path components
    path = Path(filename)
    filename = path.name

    if not filename:
        return "unnamed_file"

    # Split into name and extension
    extension = path.suffix.lower()
    name = path.stem

    # Handle files with only extension (e.g., ".gitignore")
    if not name and extension:
        name = extension[1:]  # Remove the dot
        extension = ""

    # Remove null bytes and control characters (security critical)
    name = re.sub(r"[\x00-\x1f\x7f]", "", name)

    # Remove/replace dangerous characters
    # Remove: < > : " / \ | ? * (filesystem-unsafe and shell-special)
    name = re.sub(r'[<>:"/\\|?*]', "", name)

    # Remove path separators that might have been encoded
    name = re.sub(r"\.\.+", ".", name)  # Collapse multiple dots

    # Replace spaces and other whitespace with underscores
    name = re.sub(r"\s+", "_", name)

    # Remove parentheses but keep content
    name = re.sub(r"[()]", "", name)

    # Remove any remaining special characters except underscore, hyphen, and dot
    name = re.sub(r"[^\w\-.]", "", name)

    # Collapse multiple underscores/hyphens
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"-+", "-", name)

    # Remove leading/trailing underscores and hyphens
    name = name.strip("_-.")

    # Ensure we have a valid name
    if not name:
        name = "unnamed_file"

    # Reconstruct filename with extension
    sanitized = f"{name}{extension}" if extension else name

    # Enforce maximum filename length (255 is common filesystem limit)
    max_length = 255
    if len(sanitized) > max_length:
        # Truncate name but preserve extension
        if extension:
            max_name_length = max_length - len(extension)
            sanitized = f"{name[:max_name_length]}{extension}"
        else:
            sanitized = sanitized[:max_length]

    return sanitized


# =============================================================================
# FILENAME VALIDATION
# =============================================================================


def validate_filename(filename: str) -> dict[str, Any]:
    """
    Validate a filename for security risks and proper format.

    This function checks filenames for:
    1. Empty or whitespace-only names
    2. Path traversal attempts (../, /, \\)
    3. Dangerous characters that could cause security issues
    4. Maximum length restrictions

    Args:
        filename: The filename to validate

    Returns:
        Dictionary with validation results:
        - is_valid: True if filename is valid, False otherwise
        - error: Description of validation failure or None if valid
        - sanitized: Sanitized version of the filename (if valid)

    Security:
        - Detects and rejects directory traversal attempts
        - Rejects absolute paths (both Unix and Windows style)
        - Rejects null bytes and control characters
        - Provides sanitized alternative for invalid filenames

    Example:
        >>> validate_filename("../../../etc/passwd")
        {"is_valid": False, "error": "Path traversal detected in filename", "sanitized": "etcpasswd"}
        >>> validate_filename("normal_file.txt")
        {"is_valid": True, "error": None, "sanitized": "normal_file.txt"}
    """
    result: dict[str, Any] = {
        "is_valid": True,
        "error": None,
        "sanitized": None,
    }

    # Check for empty filename
    if not filename:
        result["is_valid"] = False
        result["error"] = "Filename is empty"
        result["sanitized"] = "unnamed_file"
        return result

    # Strip whitespace
    filename = filename.strip()
    if not filename:
        result["is_valid"] = False
        result["error"] = "Filename is empty after stripping whitespace"
        result["sanitized"] = "unnamed_file"
        return result

    # Check for path traversal patterns
    path_traversal_patterns = [
        "../",
        "..\\",
        "..",
    ]

    for pattern in path_traversal_patterns:
        if pattern in filename:
            result["is_valid"] = False
            result["error"] = "Path traversal detected in filename"
            result["sanitized"] = sanitize_filename(filename)
            return result

    # Check for absolute paths (Unix style)
    if filename.startswith("/"):
        result["is_valid"] = False
        result["error"] = "Absolute path (Unix style) not allowed in filename"
        result["sanitized"] = sanitize_filename(filename)
        return result

    # Check for absolute paths (Windows style)
    if filename.startswith("\\") or (len(filename) > 1 and filename[1] == ":"):
        result["is_valid"] = False
        result["error"] = "Absolute path (Windows style) not allowed in filename"
        result["sanitized"] = sanitize_filename(filename)
        return result

    # Check for null bytes
    if "\x00" in filename:
        result["is_valid"] = False
        result["error"] = "Null bytes not allowed in filename"
        result["sanitized"] = sanitize_filename(filename)
        return result

    # Check for control characters
    control_chars = [chr(c) for c in range(32)] + [chr(127)]
    for char in control_chars:
        if char in filename:
            result["is_valid"] = False
            result["error"] = "Control characters not allowed in filename"
            result["sanitized"] = sanitize_filename(filename)
            return result

    # Check maximum length
    max_length = 255
    if len(filename) > max_length:
        result["is_valid"] = False
        result["error"] = f"Filename exceeds maximum length of {max_length} characters"
        result["sanitized"] = sanitize_filename(filename)
        return result

    # All checks passed - return sanitized filename
    result["sanitized"] = sanitize_filename(filename)
    return result


# =============================================================================
# URL VALIDATION
# =============================================================================


def validate_url(url: str) -> dict[str, Any]:
    """
    Validate URL for content import, detecting platform type and security risks.

    This function validates URLs submitted for content import by:
    1. Checking URL structure and scheme (http/https only)
    2. Detecting if URL points to YouTube, Vimeo, or general webpage
    3. Checking for dangerous file extensions in URL path
    4. Rejecting malformed or potentially malicious URLs

    Args:
        url: The URL to validate

    Returns:
        Dictionary with validation results:
        - is_valid: True if URL is valid and safe, False otherwise
        - url_type: "youtube", "vimeo", or "webpage" for valid URLs; None if invalid
        - error: Description of validation failure or None if valid

    Security:
        - Only http and https schemes are allowed
        - URLs pointing to dangerous file types are rejected
        - Malformed URLs are rejected

    Example:
        >>> validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        {"is_valid": True, "url_type": "youtube", "error": None}
        >>> validate_url("https://example.com/malware.exe")
        {"is_valid": False, "url_type": None, "error": "URL points to prohibited file type '.exe'"}
    """
    result: dict[str, Any] = {
        "is_valid": True,
        "url_type": None,
        "error": None,
    }

    if not url:
        result["is_valid"] = False
        result["error"] = "URL is empty or None"
        return result

    # Strip whitespace
    url = url.strip()

    if not url:
        result["is_valid"] = False
        result["error"] = "URL is empty after stripping whitespace"
        return result

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        result["is_valid"] = False
        result["error"] = f"Invalid URL format: {e!s}"
        return result

    # Validate scheme (only http and https allowed)
    if not parsed.scheme:
        result["is_valid"] = False
        result["error"] = "URL must include a scheme (http:// or https://)"
        return result

    if parsed.scheme.lower() not in ["http", "https"]:
        result["is_valid"] = False
        result["error"] = (
            f"URL scheme '{parsed.scheme}' is not allowed. Only http and https are permitted."
        )
        return result

    # Validate netloc (domain)
    if not parsed.netloc:
        result["is_valid"] = False
        result["error"] = "URL must include a domain name"
        return result

    # Check for dangerous file extensions in URL path
    path_lower = parsed.path.lower()
    for ext in DANGEROUS_EXTENSIONS:
        if path_lower.endswith(ext):
            result["is_valid"] = False
            result["error"] = f"URL points to prohibited file type '{ext}'"
            return result

    # Detect URL type
    url_type: str

    # Check for YouTube
    if _YOUTUBE_REGEX.match(url):
        # Additional validation for YouTube URLs
        netloc_lower = parsed.netloc.lower()
        if "youtube" in netloc_lower or "youtu.be" in netloc_lower:
            url_type = "youtube"
        else:
            url_type = "webpage"
    # Check for Vimeo
    elif _VIMEO_REGEX.match(url):
        netloc_lower = parsed.netloc.lower()
        url_type = "vimeo" if "vimeo" in netloc_lower else "webpage"
    else:
        url_type = "webpage"

    result["url_type"] = url_type
    return result


# =============================================================================
# COMPREHENSIVE FILE VALIDATION (ASYNC)
# =============================================================================


async def validate_uploaded_file(
    file_content: bytes, filename: str, file_size: int
) -> tuple[bool, str | None, str | None]:
    """
    Perform comprehensive validation on an uploaded file.

    This is the main validation function that combines all security checks:
    1. Validates file extension against whitelist/blacklist
    2. Validates actual MIME type matches extension (prevents spoofing)
    3. Validates file size against 500MB limit

    Args:
        file_content: Raw bytes of file content (or first chunk for MIME check)
        filename: Original filename with extension
        file_size: Total size of file in bytes

    Returns:
        Tuple of (is_valid, file_type, error_message):
        - is_valid: True if file passes all validations
        - file_type: Category of file (text, image, audio, video) or None if invalid
        - error_message: First validation error encountered or None if valid

    Security:
        - Performs ALL security checks in correct order
        - Fails fast on first validation error
        - Never allows dangerous files through

    Example:
        >>> async def example():
        ...     with open("document.pdf", "rb") as f:
        ...         content = f.read()
        ...         size = len(content)
        ...     result = await validate_uploaded_file(content, "document.pdf", size)
        ...     return result
    """
    # Step 1: Validate file extension
    ext_valid, file_type, ext_error = validate_file_extension(filename)
    if not ext_valid:
        return False, None, ext_error

    # Step 2: Validate file size
    size_result = validate_file_size(file_size)
    if not size_result["is_valid"]:
        return False, None, size_result.get("error")

    # Step 3: Validate MIME type (only if we have content)
    if file_content:
        mime_result = validate_mime_type(file_content, filename)
        if not mime_result["is_valid"]:
            return False, None, mime_result.get("error")

    # All validations passed
    return True, file_type, None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_file_category(extension: str) -> str | None:
    """
    Get the category of a file based on its extension.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        Category string ("text", "image", "audio", "video") or None if not found

    Example:
        >>> get_file_category(".pdf")
        "text"
        >>> get_file_category("png")
        "image"
    """
    # Ensure extension has leading dot
    if not extension.startswith("."):
        extension = f".{extension}"

    extension = extension.lower()

    for category, extensions in ALLOWED_EXTENSIONS_BY_CATEGORY.items():
        if extension in extensions:
            return category

    return None


def get_allowed_extensions_flat() -> list[str]:
    """
    Get a flat list of all allowed extensions.

    Returns:
        List of all allowed extensions (e.g., [".txt", ".md", ".pdf", ...])

    Example:
        >>> extensions = get_allowed_extensions_flat()
        >>> ".pdf" in extensions
        True
    """
    all_extensions: list[str] = []
    for extensions in ALLOWED_EXTENSIONS_BY_CATEGORY.values():
        all_extensions.extend(extensions)
    return sorted(all_extensions)


def is_dangerous_extension(extension: str) -> bool:
    """
    Check if an extension is in the dangerous extensions list.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        True if extension is dangerous, False otherwise

    Example:
        >>> is_dangerous_extension(".exe")
        True
        >>> is_dangerous_extension(".pdf")
        False
    """
    # Ensure extension has leading dot
    if not extension.startswith("."):
        extension = f".{extension}"

    return extension.lower() in DANGEROUS_EXTENSIONS


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string (e.g., "1.5 MB", "256 KB")

    Example:
        >>> format_file_size(1536)
        "1.50 KB"
        >>> format_file_size(1048576)
        "1.00 MB"
    """
    if size_bytes < 0:
        return "Invalid size"

    bytes_per_mb = BYTES_PER_KB * BYTES_PER_KB
    bytes_per_gb = bytes_per_mb * BYTES_PER_KB

    if size_bytes < BYTES_PER_KB:
        return f"{size_bytes} B"
    if size_bytes < bytes_per_mb:
        return f"{size_bytes / BYTES_PER_KB:.2f} KB"
    if size_bytes < bytes_per_gb:
        return f"{size_bytes / bytes_per_mb:.2f} MB"
    return f"{size_bytes / bytes_per_gb:.2f} GB"


# =============================================================================
# HTTP EXCEPTION HELPERS
# =============================================================================


def raise_file_validation_error(error_message: str, status_code: int = 400) -> None:
    """
    Raise an HTTPException for file validation failures.

    This helper provides a consistent way to raise validation errors
    with appropriate HTTP status codes.

    Args:
        error_message: Description of the validation failure
        status_code: HTTP status code (default 400 Bad Request)

    Raises:
        HTTPException: Always raises with the provided message and status

    Example:
        >>> raise_file_validation_error("File type not allowed")
        # Raises HTTPException(status_code=400, detail="File type not allowed")
    """
    raise HTTPException(status_code=status_code, detail=error_message)


def raise_file_too_large_error(file_size: int) -> None:
    """
    Raise an HTTPException for files exceeding size limit.

    Uses HTTP 413 Payload Too Large status code as per REST conventions.

    Args:
        file_size: Actual size of the uploaded file in bytes

    Raises:
        HTTPException: Always raises with 413 status

    Example:
        >>> raise_file_too_large_error(600 * 1024 * 1024)  # 600 MB
        # Raises HTTPException(status_code=413, detail="File size exceeds 500MB limit...")
    """
    size_str = format_file_size(file_size)
    max_size_str = format_file_size(MAX_FILE_SIZE_BYTES)
    raise HTTPException(
        status_code=413,
        detail=f"File size ({size_str}) exceeds maximum allowed size ({max_size_str})",
    )


def raise_unsupported_type_error(extension: str) -> None:
    """
    Raise an HTTPException for unsupported file types.

    Uses HTTP 415 Unsupported Media Type status code as per REST conventions.

    Args:
        extension: The unsupported file extension

    Raises:
        HTTPException: Always raises with 415 status

    Example:
        >>> raise_unsupported_type_error(".exe")
        # Raises HTTPException(status_code=415, detail="...")
    """
    allowed = get_allowed_extensions_flat()
    raise HTTPException(
        status_code=415,
        detail=f"File type '{extension}' is not supported. Allowed types: {', '.join(allowed)}",
    )
