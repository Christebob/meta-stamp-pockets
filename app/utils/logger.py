"""
Structured Logging Configuration Module for META-STAMP V3

This module provides comprehensive logging utilities with JSON-formatted output,
log rotation, context enrichment via LoggerAdapter, and integration with Uvicorn's
logger for consistent application-wide logging across all backend services.

Features:
- JSONFormatter: Custom formatter outputting structured JSON log records
- get_logger: Factory function for creating configured loggers
- setup_logging: Application-wide logging configuration with Uvicorn integration
- add_log_context: Helper for enriching logs with contextual information

Usage:
    from app.utils.logger import get_logger, setup_logging, add_log_context

    # Initialize logging at application startup
    setup_logging(log_level="INFO", json_logs=True)

    # Get a logger for a specific module
    logger = get_logger(__name__)
    logger.info("Application started")

    # Add context to logs (e.g., request_id, user_id)
    ctx_logger = add_log_context(logger, request_id="abc123", user_id="user456")
    ctx_logger.info("Processing request")

Per Agent Action Plan sections 0.4 (Implementation Design), 0.6 (Utilities Layer),
and 0.10 (Execution Parameters requiring structured logging).
"""

import json
import logging
import logging.handlers
import os
import sys
import traceback

from collections.abc import MutableMapping
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar


# =============================================================================
# Constants
# =============================================================================

# Default log directory for file-based logging
DEFAULT_LOG_DIR: str = "logs"

# Log rotation settings per Agent Action Plan section 0.6
# 10 MB per file, keep 5 backup files
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT: int = 5

# Default log filename
DEFAULT_LOG_FILENAME: str = "meta_stamp.log"

# Log level mapping from string to logging constants
LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Third-party loggers to reduce verbosity
THIRD_PARTY_LOGGERS: list[str] = [
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "fastapi",
    "motor",
    "pymongo",
    "boto3",
    "botocore",
    "s3transfer",
    "urllib3",
    "httpx",
    "httpcore",
    "langchain",
    "openai",
    "anthropic",
    "redis",
    "asyncio",
]


# =============================================================================
# Custom JSON Encoder
# =============================================================================


class LogJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for log record serialization.

    Handles non-serializable objects by converting them to string representations,
    ensuring that log records can always be serialized to JSON format even when
    containing complex Python objects.
    """

    def default(self, obj: Any) -> Any:
        """
        Override default serialization for non-standard types.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Exception):
            return str(obj)
        if hasattr(obj, "__dict__"):
            try:
                return str(obj)
            except Exception:
                return f"<non-serializable: {type(obj).__name__}>"
        try:
            return str(obj)
        except Exception:
            return f"<non-serializable: {type(obj).__name__}>"


# =============================================================================
# JSONFormatter Class
# =============================================================================


class JSONFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs log records as JSON strings.

    This formatter converts Python logging LogRecord objects into structured
    JSON format, including timestamp, log level, logger name, message, and
    any extra fields provided via LoggerAdapter or the extra parameter.

    The JSON format enables easy parsing by log aggregation systems (e.g.,
    Elasticsearch, Splunk, CloudWatch) and provides consistent structure
    for log analysis and debugging across distributed services.

    Attributes:
        include_extra_fields: Whether to include extra context fields in output
        datefmt: Date format string (not used, ISO 8601 is always used)

    Example output:
        {
            "timestamp": "2025-01-15T10:30:45.123456Z",
            "level": "INFO",
            "logger": "app.services.upload_service",
            "message": "File uploaded successfully",
            "request_id": "abc123",
            "user_id": "user456",
            "file_name": "document.pdf"
        }
    """

    # Standard LogRecord attributes to exclude from extra fields
    RESERVED_ATTRS: ClassVar[set[str]] = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }

    def __init__(
        self,
        include_extra_fields: bool = True,
        include_source_location: bool = False,
        include_process_info: bool = False,
    ) -> None:
        """
        Initialize the JSON formatter.

        Args:
            include_extra_fields: If True, include custom extra fields from
                                  LoggerAdapter or extra dict in log output
            include_source_location: If True, include filename, lineno, funcName
            include_process_info: If True, include process and thread information
        """
        super().__init__()
        self.include_extra_fields = include_extra_fields
        self.include_source_location = include_source_location
        self.include_process_info = include_process_info

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a LogRecord as a JSON string.

        Converts the LogRecord to a dictionary containing timestamp, level,
        logger name, message, exception info (if present), and any extra
        context fields. The dictionary is then serialized to JSON.

        Args:
            record: The LogRecord to format

        Returns:
            JSON-formatted string representation of the log record
        """
        # Build base log entry with core fields
        log_entry: dict[str, Any] = {
            "timestamp": self._format_timestamp(record),
            "level": record.levelname,
            "logger": record.name,
            "message": self._format_message(record),
        }

        # Include source location if requested (useful for debugging)
        if self.include_source_location:
            log_entry["source"] = {
                "filename": record.filename,
                "lineno": record.lineno,
                "function": record.funcName,
            }

        # Include process/thread info if requested (useful for concurrent debugging)
        if self.include_process_info:
            log_entry["process"] = {
                "id": record.process,
                "name": record.processName,
                "thread_id": record.thread,
                "thread_name": record.threadName,
            }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self._format_exception(record)

        # Add stack info if present
        if record.stack_info:
            log_entry["stack_info"] = record.stack_info

        # Include extra fields from LoggerAdapter or extra parameter
        if self.include_extra_fields:
            extra_fields = self._extract_extra_fields(record)
            if extra_fields:
                log_entry["extra"] = extra_fields

        # Serialize to JSON with custom encoder for non-standard types
        try:
            return json.dumps(
                log_entry,
                cls=LogJSONEncoder,
                ensure_ascii=False,
                separators=(",", ":"),  # Compact format
            )
        except (TypeError, ValueError) as e:
            # Fallback if JSON serialization fails
            fallback_entry = {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": "ERROR",
                "logger": "JSONFormatter",
                "message": f"Failed to serialize log record: {e}",
                "original_message": str(record.msg),
            }
            return json.dumps(fallback_entry, ensure_ascii=False)

    def _format_timestamp(self, record: logging.LogRecord) -> str:
        """
        Format the log record timestamp in ISO 8601 format with UTC timezone.

        Args:
            record: The LogRecord containing the creation time

        Returns:
            ISO 8601 formatted timestamp string with 'Z' suffix for UTC
        """
        # Convert record creation time to datetime
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        return dt.isoformat(timespec="microseconds")

    def _format_message(self, record: logging.LogRecord) -> str:
        """
        Format the log message, applying any arguments if present.

        Args:
            record: The LogRecord containing the message

        Returns:
            Formatted message string
        """
        try:
            return record.getMessage()
        except Exception:
            # Fallback if message formatting fails
            return str(record.msg)

    def _format_exception(self, record: logging.LogRecord) -> dict[str, Any]:
        """
        Format exception information into a structured dictionary.

        Args:
            record: The LogRecord containing exception info

        Returns:
            Dictionary with exception type, message, and formatted traceback
        """
        exc_info = record.exc_info
        if exc_info:
            exc_type, exc_value, _exc_tb = exc_info
            return {
                "type": exc_type.__name__ if exc_type else "Unknown",
                "message": str(exc_value) if exc_value else "",
                "traceback": self._format_traceback(exc_info),
            }
        return {}

    def _format_traceback(
        self,
        exc_info: tuple[type[BaseException] | None, BaseException | None, TracebackType | None],
    ) -> str:
        """
        Format the exception traceback as a string.

        Args:
            exc_info: Exception info tuple (type, value, traceback)

        Returns:
            Formatted traceback string
        """
        try:
            return "".join(traceback.format_exception(*exc_info))
        except Exception:
            return ""

    def _extract_extra_fields(self, record: logging.LogRecord) -> dict[str, Any]:
        """
        Extract extra fields from the log record that aren't standard attributes.

        This allows LoggerAdapter extra context and logger.info(..., extra={})
        parameters to be included in the JSON output.

        Args:
            record: The LogRecord to extract extra fields from

        Returns:
            Dictionary of extra field names and values
        """
        extra_fields: dict[str, Any] = {}

        for key, value in record.__dict__.items():
            # Skip private attributes and reserved LogRecord attributes
            if key.startswith("_") or key in self.RESERVED_ATTRS:
                continue
            extra_fields[key] = value

        return extra_fields


# =============================================================================
# Standard Text Formatter
# =============================================================================


class StandardFormatter(logging.Formatter):
    """
    Standard text formatter for console output in development mode.

    Provides human-readable log output with timestamp, level, logger name,
    and message. Useful for local development where JSON format may be
    harder to read.

    Format: [TIMESTAMP] LEVEL logger_name: message
    """

    DEFAULT_FORMAT: str = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
    DEFAULT_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
    ) -> None:
        """
        Initialize the standard formatter.

        Args:
            fmt: Log message format string (uses DEFAULT_FORMAT if not provided)
            datefmt: Date format string (uses DEFAULT_DATE_FORMAT if not provided)
        """
        super().__init__(
            fmt=fmt or self.DEFAULT_FORMAT,
            datefmt=datefmt or self.DEFAULT_DATE_FORMAT,
        )


# =============================================================================
# Logger Factory Function
# =============================================================================


def get_logger(
    name: str,
    level: str | None = None,
    json_logs: bool = True,
    include_file_handler: bool = False,
    log_dir: str | None = None,
    log_filename: str | None = None,
) -> logging.Logger:
    """
    Create and configure a logger with the specified name and settings.

    This factory function creates a logger with appropriate handlers and
    formatters based on the configuration. It prevents duplicate handler
    registration by checking for existing handlers.

    Args:
        name: Logger name, typically __name__ of the calling module
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to environment variable LOG_LEVEL or "INFO"
        json_logs: If True, use JSONFormatter; if False, use StandardFormatter
        include_file_handler: If True, add RotatingFileHandler for file logging
        log_dir: Directory for log files (defaults to DEFAULT_LOG_DIR)
        log_filename: Log filename (defaults to DEFAULT_LOG_FILENAME)

    Returns:
        Configured logging.Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Application started")

        # With custom settings
        logger = get_logger(
            "my_service",
            level="DEBUG",
            json_logs=True,
            include_file_handler=True
        )
    """
    # Get or create logger
    logger = logging.getLogger(name)

    # Determine log level from parameter, environment, or default
    level = os.environ.get("LOG_LEVEL", "INFO").upper() if level is None else level.upper()

    log_level = LOG_LEVEL_MAP.get(level, logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Prevent log messages from propagating to root logger
    # (this prevents duplicate messages when root is configured)
    logger.propagate = False

    # Select formatter based on json_logs setting
    if json_logs:
        formatter: logging.Formatter = JSONFormatter(
            include_extra_fields=True,
            include_source_location=log_level <= logging.DEBUG,
        )
    else:
        formatter = StandardFormatter()

    # Add console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if requested
    if include_file_handler:
        file_handler = _create_file_handler(
            log_dir=log_dir or DEFAULT_LOG_DIR,
            log_filename=log_filename or DEFAULT_LOG_FILENAME,
            log_level=log_level,
            formatter=formatter,
        )
        if file_handler:
            logger.addHandler(file_handler)

    return logger


def _create_file_handler(
    log_dir: str,
    log_filename: str,
    log_level: int,
    formatter: logging.Formatter,
) -> RotatingFileHandler | None:
    """
    Create a RotatingFileHandler for file-based logging with rotation.

    Creates the log directory if it doesn't exist and configures rotation
    with 10MB max file size and 5 backup files.

    Args:
        log_dir: Directory path for log files
        log_filename: Name of the log file
        log_level: Logging level for the handler
        formatter: Formatter to apply to the handler

    Returns:
        Configured RotatingFileHandler or None if creation fails
    """
    try:
        # Create log directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        log_path = log_dir_path / log_filename

        # Create rotating file handler with configured limits
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        return file_handler

    except (OSError, PermissionError) as e:
        # Log to stderr if file handler creation fails
        sys.stderr.write(f"Warning: Could not create log file handler: {e}\n")
        return None


# =============================================================================
# Application Logging Setup
# =============================================================================


def setup_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    include_file_logging: bool = False,
    log_dir: str | None = None,
    log_filename: str | None = None,
    third_party_level: str = "WARNING",
) -> None:
    """
    Configure application-wide logging with root logger and Uvicorn integration.

    This function should be called once at application startup (typically in
    main.py or during FastAPI lifespan initialization). It configures:
    - Root logger with appropriate handlers and formatters
    - Console output with JSON or standard formatting
    - Optional file output with rotation
    - Uvicorn logger integration for consistent request logging
    - Third-party logger level reduction for less verbose output

    Args:
        log_level: Application log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON format; if False, output standard text
        include_file_logging: If True, add file handler with rotation
        log_dir: Directory for log files (defaults to "logs")
        log_filename: Log filename (defaults to "meta_stamp.log")
        third_party_level: Log level for third-party libraries (default WARNING)

    Example:
        # In main.py or FastAPI lifespan
        from app.utils.logger import setup_logging

        # Development: human-readable logs
        setup_logging(log_level="DEBUG", json_logs=False)

        # Production: JSON logs with file output
        setup_logging(
            log_level="INFO",
            json_logs=True,
            include_file_logging=True
        )
    """
    # Get log level from string
    level_str = log_level.upper()
    level = LOG_LEVEL_MAP.get(level_str, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter based on json_logs setting
    if json_logs:
        formatter: logging.Formatter = JSONFormatter(
            include_extra_fields=True,
            include_source_location=level <= logging.DEBUG,
        )
    else:
        formatter = StandardFormatter()

    # Add console handler to root logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if requested
    if include_file_logging:
        file_handler = _create_file_handler(
            log_dir=log_dir or DEFAULT_LOG_DIR,
            log_filename=log_filename or DEFAULT_LOG_FILENAME,
            log_level=level,
            formatter=formatter,
        )
        if file_handler:
            root_logger.addHandler(file_handler)

    # Configure Uvicorn loggers for consistent output
    _configure_uvicorn_logging(formatter, level)

    # Reduce verbosity of third-party loggers
    third_party_log_level = LOG_LEVEL_MAP.get(third_party_level.upper(), logging.WARNING)
    _configure_third_party_loggers(third_party_log_level)

    # Log setup completion
    setup_logger = logging.getLogger("app.utils.logger")
    setup_logger.info(
        f"Logging configured: level={level_str}, "
        f"json={json_logs}, "
        f"file_logging={include_file_logging}"
    )


def _configure_uvicorn_logging(
    formatter: logging.Formatter,
    level: int,
) -> None:
    """
    Configure Uvicorn loggers to use consistent formatting with application.

    Integrates Uvicorn's access and error loggers with the application's
    logging configuration for uniform log output format.

    Args:
        formatter: Formatter to apply to Uvicorn handlers
        level: Log level to set for Uvicorn loggers
    """
    # Configure Uvicorn's main logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(level)
    uvicorn_logger.propagate = False

    # Clear existing handlers and add console with our formatter
    uvicorn_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    uvicorn_logger.addHandler(handler)

    # Configure Uvicorn access logger
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(level)
    uvicorn_access_logger.propagate = False
    uvicorn_access_logger.handlers.clear()
    access_handler = logging.StreamHandler(sys.stdout)
    access_handler.setFormatter(formatter)
    uvicorn_access_logger.addHandler(access_handler)

    # Configure Uvicorn error logger
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(level)
    uvicorn_error_logger.propagate = False
    uvicorn_error_logger.handlers.clear()
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setFormatter(formatter)
    uvicorn_error_logger.addHandler(error_handler)


def _configure_third_party_loggers(level: int) -> None:
    """
    Set log levels for third-party libraries to reduce noise.

    Many third-party libraries produce verbose DEBUG/INFO logs that
    can overwhelm application logs. This sets them to WARNING or above.

    Args:
        level: Log level to set for third-party loggers
    """
    for logger_name in THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(level)


# =============================================================================
# Context Enrichment
# =============================================================================


class ContextLoggerAdapter(logging.LoggerAdapter[logging.Logger]):
    """
    Custom LoggerAdapter that properly merges context with log record extra fields.

    Extends the standard LoggerAdapter to ensure that context fields are
    properly merged with any extra fields passed during logging calls,
    rather than overwriting them.
    """

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        """
        Process the logging call by merging context with extra fields.

        Args:
            msg: The log message
            kwargs: Keyword arguments passed to the logging call

        Returns:
            Tuple of (message, modified kwargs)
        """
        # Get existing extra dict or create empty one
        extra: dict[str, Any] = dict(kwargs.get("extra", {}))

        # Merge our context into extra, preserving existing values
        if self.extra is not None:
            for key, value in self.extra.items():
                if key not in extra:
                    extra[key] = value

        kwargs["extra"] = extra
        return msg, kwargs


def add_log_context(
    logger: logging.Logger,
    **kwargs: Any,
) -> logging.LoggerAdapter[logging.Logger]:
    """
    Create a LoggerAdapter that enriches all log messages with context fields.

    This helper function wraps a logger with contextual information that will
    be automatically included in every log message. Useful for adding:
    - request_id: Unique identifier for request tracing
    - user_id: ID of the authenticated user
    - asset_id: ID of the asset being processed
    - operation: Name of the current operation

    Args:
        logger: Base logger to wrap with context
        **kwargs: Context fields to include in all log messages

    Returns:
        LoggerAdapter that includes the specified context in all log output

    Example:
        logger = get_logger(__name__)

        # In request handler:
        ctx_logger = add_log_context(
            logger,
            request_id="abc-123",
            user_id="user-456",
            operation="upload_file"
        )

        ctx_logger.info("Starting file upload")
        # Output includes: "request_id": "abc-123", "user_id": "user-456", ...

        ctx_logger.error("Upload failed", extra={"file_size": 1024})
        # Output includes all context plus file_size
    """
    return ContextLoggerAdapter(logger, kwargs)


# =============================================================================
# Utility Functions
# =============================================================================


def get_log_level_from_string(level_str: str) -> int:
    """
    Convert a log level string to its corresponding logging constant.

    Args:
        level_str: String representation of log level
                   (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Corresponding logging level constant (defaults to INFO if invalid)
    """
    return LOG_LEVEL_MAP.get(level_str.upper(), logging.INFO)


def set_log_level(logger_name: str, level: str) -> None:
    """
    Dynamically update the log level for a specific logger.

    Useful for runtime log level adjustments (e.g., via admin API).

    Args:
        logger_name: Name of the logger to modify
        level: New log level string
    """
    target_logger = logging.getLogger(logger_name)
    new_level = LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
    target_logger.setLevel(new_level)

    # Also update handler levels
    for handler in target_logger.handlers:
        handler.setLevel(new_level)


def get_all_loggers() -> dict[str, dict[str, Any]]:
    """
    Get information about all configured loggers.

    Useful for debugging logging configuration and monitoring.

    Returns:
        Dictionary mapping logger names to their configuration info
    """
    loggers_info: dict[str, dict[str, Any]] = {}

    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            loggers_info[name] = {
                "level": logging.getLevelName(logger.level),
                "effective_level": logging.getLevelName(logger.getEffectiveLevel()),
                "handlers": [type(h).__name__ for h in logger.handlers],
                "propagate": logger.propagate,
            }

    return loggers_info


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILENAME",
    "LOG_BACKUP_COUNT",
    "LOG_LEVEL_MAP",
    "LOG_MAX_BYTES",
    "ContextLoggerAdapter",
    "JSONFormatter",
    "StandardFormatter",
    "add_log_context",
    "get_all_loggers",
    "get_log_level_from_string",
    "get_logger",
    "set_log_level",
    "setup_logging",
]
