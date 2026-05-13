"""
FastAPI Router for AI Assistant Endpoint - META-STAMP V3.

This module implements the AI assistant endpoint POST /ask with streaming Server-Sent Events
response using LangChain multi-provider support (OpenAI GPT-4/5, Anthropic Claude, Google
Gemini, local models), tool calling capabilities for querying fingerprint data and analytics,
conversation context management in Redis, and hybrid-personality modes (friendly guidance
and serious legal advisory).

Per Agent Action Plan:
- Section 0.1: Multi-provider AI assistant requirement
- Section 0.3: LangChain integration with OpenAI, Anthropic, Google providers
- Section 0.4: Streaming responses using Server-Sent Events
- Section 0.6: Assistant endpoint transformation mapping
- Section 0.8: Streaming endpoint implementation with Redis context (1-hour TTL)
- Section 0.10: Execution parameters for AI integration

Endpoints:
    POST /ask - Send query to AI assistant with streaming Server-Sent Events response

Features:
    - Multi-provider LLM support via LangChain
    - Real-time streaming responses using SSE
    - Tool calling for fingerprint and analytics queries
    - Conversation context persistence in Redis (1-hour TTL)
    - Dual personality modes: friendly guidance and legal advisory
    - Comprehensive error handling with appropriate HTTP status codes

Author: META-STAMP V3 Platform
License: Proprietary
"""

import json
import logging
import uuid

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.auth import get_current_user
from app.core.redis_client import get_redis_client
from app.services.ai_assistant_service import (
    AIAssistantService,
    ProviderInitializationError,
    StreamingError,
)
from app.services.ai_value_service import AIValueService
from app.services.fingerprinting_service import FingerprintingService


# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(tags=["assistant"])

# =============================================================================
# Constants
# =============================================================================

# Supported AI providers per Agent Action Plan section 0.3
SUPPORTED_PROVIDERS = ["openai", "anthropic", "google", "local"]

# Supported personality modes per Agent Action Plan section 0.1
SUPPORTED_PERSONALITIES = ["friendly", "legal"]

# Default model configurations for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4",
    "anthropic": "claude-3-5-sonnet-latest",
    "google": "gemini-2.0-flash",
    "local": "llama3.2",
}

# SSE event formatting
SSE_DATA_PREFIX = "data: "
SSE_EVENT_SUFFIX = "\n\n"

# Maximum message length per request
MAX_MESSAGE_LENGTH = 2000
MIN_MESSAGE_LENGTH = 1


# =============================================================================
# Request/Response Models
# =============================================================================


class AskRequest(BaseModel):
    """
    Request model for AI assistant /ask endpoint.

    Validates and structures the user's message along with optional
    configuration parameters for provider selection and personality mode.

    Attributes:
        message: The user's question or statement (1-2000 characters)
        conversation_id: Optional existing conversation ID for context continuity
        provider: Optional AI provider override (openai, anthropic, google, local)
        personality: Personality mode - "friendly" for general help,
                    "legal" for legal advisory mode

    Example:
        ```json
        {
            "message": "What is the fingerprint status of my latest upload?",
            "conversation_id": "conv_abc123",
            "provider": "openai",
            "personality": "friendly"
        }
        ```
    """

    message: str = Field(
        ...,
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH,
        description="The user's message to the AI assistant (1-2000 characters)",
        json_schema_extra={"example": "What is my AI Touch Value for asset X?"},
    )

    conversation_id: str | None = Field(
        default=None,
        max_length=100,
        description="Optional conversation ID for context continuity",
        json_schema_extra={"example": "conv_abc123def456"},
    )

    provider: str | None = Field(
        default=None,
        description="AI provider: openai, anthropic, google, or local",
        json_schema_extra={"example": "openai"},
    )

    personality: str | None = Field(
        default="friendly",
        description="Personality mode: friendly (general help) or legal (advisory)",
        json_schema_extra={"example": "friendly"},
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Can you explain my fingerprint data?",
                "conversation_id": None,
                "provider": "openai",
                "personality": "friendly",
            }
        }
    }


class ToolCallInfo(BaseModel):
    """
    Information about a tool call made by the AI assistant.

    When the assistant invokes tools like fingerprint lookup or analytics query,
    this model captures the tool invocation details for client visibility.

    Attributes:
        name: The tool function name that was invoked
        args: Arguments passed to the tool function
        result: The result returned by the tool (if available)
    """

    name: str = Field(..., description="Tool function name")
    args: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    result: dict[str, Any] | None = Field(default=None, description="Tool result")


class MessageResponse(BaseModel):
    """
    Response model for non-streaming message responses.

    Used for returning the complete assistant response along with metadata
    about the interaction, including any tool calls that were made.

    Attributes:
        content: The assistant's response text
        role: Message role (always "assistant" for responses)
        timestamp: UTC timestamp when response was generated
        tool_calls: List of tool calls made during response generation
        conversation_id: The conversation ID for context continuity
        provider: The AI provider that generated the response
        personality: The personality mode used for generation
    """

    content: str = Field(..., description="The assistant's response text")
    role: str = Field(default="assistant", description="Message role")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Response generation timestamp (UTC)",
    )
    tool_calls: list[ToolCallInfo] | None = Field(
        default=None, description="Tool calls made during response generation"
    )
    conversation_id: str = Field(..., description="Conversation ID for continuity")
    provider: str = Field(..., description="AI provider used")
    personality: str = Field(..., description="Personality mode used")

    model_config = {
        "json_encoders": {datetime: lambda v: v.isoformat()},
        "json_schema_extra": {
            "example": {
                "content": "Your asset has been fingerprinted successfully...",
                "role": "assistant",
                "timestamp": "2024-01-15T10:30:00Z",
                "tool_calls": None,
                "conversation_id": "conv_abc123",
                "provider": "openai",
                "personality": "friendly",
            }
        },
    }


class StreamChunk(BaseModel):
    """
    Model for individual stream chunks in SSE response.

    Each chunk represents a single event in the Server-Sent Events stream,
    containing either a content token, a tool call notification, or a
    completion signal.

    Chunk Types:
        - "token": A content token (word/character) from the AI response
        - "tool_call": Notification that a tool was invoked
        - "tool_result": Result from a tool execution
        - "done": Signals completion of the response stream
        - "error": Error occurred during streaming

    Attributes:
        type: The chunk type indicating what kind of data this represents
        content: Text content (for token type)
        name: Tool name (for tool_call type)
        args: Tool arguments (for tool_call type)
        result: Tool result (for tool_result type)
        message_id: Unique message identifier (for done type)
        error: Error message (for error type)
    """

    type: str = Field(..., description="Chunk type: token, tool_call, tool_result, done, error")
    content: str | None = Field(default=None, description="Content for token chunks")
    name: str | None = Field(default=None, description="Tool name for tool_call chunks")
    args: dict[str, Any] | None = Field(default=None, description="Tool arguments")
    result: dict[str, Any] | None = Field(default=None, description="Tool result")
    message_id: str | None = Field(default=None, description="Message ID for done chunk")
    error: str | None = Field(default=None, description="Error message for error chunks")


# =============================================================================
# Helper Functions
# =============================================================================


def _validate_provider(provider: str | None) -> str:
    """
    Validate and normalize the AI provider selection.

    Args:
        provider: The requested provider name or None for default

    Returns:
        str: Validated provider name

    Raises:
        HTTPException: With 400 status if provider is not supported
    """
    if provider is None:
        return "openai"  # Default provider

    provider_lower = provider.lower().strip()

    if provider_lower not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_provider",
                "message": f"Provider '{provider}' is not supported",
                "supported_providers": SUPPORTED_PROVIDERS,
            },
        )

    return provider_lower


def _validate_personality(personality: str | None) -> str:
    """
    Validate and normalize the personality mode selection.

    Args:
        personality: The requested personality mode or None for default

    Returns:
        str: Validated personality mode

    Raises:
        HTTPException: With 400 status if personality is not supported
    """
    if personality is None:
        return "friendly"  # Default personality

    personality_lower = personality.lower().strip()

    if personality_lower not in SUPPORTED_PERSONALITIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_personality",
                "message": f"Personality '{personality}' is not supported",
                "supported_personalities": SUPPORTED_PERSONALITIES,
            },
        )

    return personality_lower


def _format_sse_event(data: dict[str, Any]) -> str:
    """
    Format a dictionary as a Server-Sent Events data line.

    Args:
        data: The data dictionary to format as JSON

    Returns:
        str: Formatted SSE data line with proper line endings
    """
    json_data = json.dumps(data, default=str)
    return f"{SSE_DATA_PREFIX}{json_data}{SSE_EVENT_SUFFIX}"


def _get_or_create_conversation_id(request_conversation_id: str | None) -> str:
    """
    Get existing conversation ID or generate a new one.

    Args:
        request_conversation_id: Conversation ID from request, or None

    Returns:
        str: Existing or newly generated conversation ID
    """
    if request_conversation_id:
        return request_conversation_id

    # Generate new conversation ID with prefix for identification
    new_id = f"conv_{uuid.uuid4().hex[:16]}"
    logger.debug("Generated new conversation ID: %s", new_id)
    return new_id


def _create_ai_assistant_service(settings: Settings) -> AIAssistantService:
    """
    Create and configure an AIAssistantService instance.

    Initializes the service with fingerprinting and AI value service dependencies,
    and configures API keys from application settings.

    Args:
        settings: Application settings containing API keys

    Returns:
        AIAssistantService: Configured assistant service instance
    """
    # Create dependent services
    fingerprint_service = FingerprintingService()
    ai_value_service = AIValueService()

    # Create assistant service with provider API keys
    return AIAssistantService(
        fingerprint_service=fingerprint_service,
        ai_value_service=ai_value_service,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        google_api_key=settings.google_api_key,
        default_provider="openai",
        default_model="gpt-4",
    )


# =============================================================================
# Streaming Response Generator
# =============================================================================


async def stream_assistant_response(
    assistant_service: AIAssistantService,
    user_id: str,
    conversation_id: str,
    message: str,
    provider: str,
    model: str,
    personality: str,
) -> AsyncIterator[str]:
    """
    Async generator that streams AI assistant response as SSE events.

    This generator yields Server-Sent Events formatted chunks as the AI model
    generates its response. Each chunk contains either a content token,
    a tool call notification, or a completion signal.

    SSE Event Format:
        - Token: {"type": "token", "content": "word"}
        - Tool call: {"type": "tool_call", "name": "...", "args": {...}}
        - Tool result: {"type": "tool_result", "name": "...", "result": {...}}
        - Done: {"type": "done", "message_id": "...", "conversation_id": "..."}
        - Error: {"type": "error", "error": "..."}

    Args:
        assistant_service: Configured AIAssistantService instance
        user_id: Authenticated user's ID
        conversation_id: Conversation ID for context management
        message: User's message text
        provider: Selected AI provider
        model: Model name for the provider
        personality: Personality mode (friendly/legal)

    Yields:
        str: SSE formatted event strings

    Note:
        The generator handles errors gracefully, yielding error events
        rather than raising exceptions to maintain the SSE connection.
    """
    message_id = f"msg_{uuid.uuid4().hex[:12]}"
    logger.info(
        "Starting streaming response: message_id=%s, user=%s, conv=%s, provider=%s",
        message_id,
        user_id,
        conversation_id,
        provider,
    )

    try:
        # Stream response from AI assistant service
        async for chunk in assistant_service.send_message_stream(
            user_id=user_id,
            conversation_id=conversation_id,
            message=message,
            provider=provider,
            model=model,
            personality=personality,
        ):
            # Each chunk from the service is a content string
            if chunk:
                yield _format_sse_event({"type": "token", "content": chunk})

        # Send completion event
        yield _format_sse_event(
            {
                "type": "done",
                "message_id": message_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        logger.info("Streaming completed: message_id=%s", message_id)

    except ProviderInitializationError as e:
        logger.exception("Provider initialization failed")
        yield _format_sse_event(
            {
                "type": "error",
                "error": f"Provider initialization failed: {e!s}",
                "code": "provider_error",
            }
        )

    except StreamingError as e:
        logger.exception("Streaming error")
        yield _format_sse_event(
            {
                "type": "error",
                "error": f"Streaming error: {e!s}",
                "code": "streaming_error",
            }
        )

    except Exception:
        logger.exception("Unexpected error during streaming")
        yield _format_sse_event(
            {
                "type": "error",
                "error": "An unexpected error occurred",
                "code": "internal_error",
            }
        )


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/ask",
    response_class=StreamingResponse,
    summary="Ask AI Assistant",
    description="""
Send a query to the META-STAMP V3 AI assistant and receive a streaming response.

## Overview

The AI assistant helps creators understand their assets, fingerprints, AI Touch Values,
and provides guidance on the platform. It supports multiple AI providers and can operate
in different personality modes.

## Streaming Response

Responses are streamed using Server-Sent Events (SSE) format. Each event is a JSON
object with a `type` field indicating the event type:

- `token`: Content chunk from the AI response
- `tool_call`: Notification that a tool is being invoked
- `tool_result`: Result from a tool execution
- `done`: Stream completion signal
- `error`: Error occurred during streaming

## Providers

The following AI providers are supported:
- **openai**: GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **anthropic**: Claude-3-opus, Claude-3-sonnet, Claude-3-haiku
- **google**: Gemini-pro, Gemini-pro-vision
- **local**: Ollama-based local models

## Personality Modes

- **friendly**: Warm, helpful guidance for general platform questions
- **legal**: Serious, professional tone for legal and compensation topics

## Conversation Context

Conversations are maintained in Redis with a 1-hour TTL. Provide a `conversation_id`
to continue an existing conversation, or omit it to start a new one.

## Tool Capabilities

The assistant can invoke tools to:
- Look up fingerprint data for your assets
- Query AI Touch Value analytics and calculations
""",
    responses={
        200: {
            "description": "Streaming response with Server-Sent Events",
            "content": {
                "text/event-stream": {
                    "example": 'data: {"type": "token", "content": "Hello"}\n\ndata: {"type": "done", "message_id": "msg_abc123"}\n\n'
                }
            },
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "error": "invalid_provider",
                            "message": "Provider 'invalid' is not supported",
                            "supported_providers": ["openai", "anthropic", "google", "local"],
                        }
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
            "content": {"application/json": {"example": {"detail": "Invalid token"}}},
        },
        429: {
            "description": "Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {"detail": "Rate limit exceeded. Please try again later."}
                }
            },
        },
        503: {
            "description": "AI provider unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "AI provider is currently unavailable. Please try again later."
                    }
                }
            },
        },
    },
    tags=["assistant"],
)
async def ask_assistant(
    request: AskRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Process a user message and stream the AI assistant's response.

    This endpoint accepts a user message, processes it through the configured
    AI provider with the selected personality mode, and streams the response
    back using Server-Sent Events.

    The endpoint requires authentication via JWT token. Conversation context
    is automatically managed in Redis with a 1-hour TTL.

    Args:
        request: The user's message and configuration options
        current_user: Authenticated user from JWT token (injected)
        settings: Application settings (injected)

    Returns:
        StreamingResponse: SSE stream with AI assistant response chunks

    Raises:
        HTTPException: With appropriate status code on validation or processing errors
    """
    # Extract user information
    user_id = current_user.get("_id") or current_user.get("id") or "unknown"
    user_email = current_user.get("email", "unknown")

    logger.info(
        "AI Assistant request: user=%s (%s), provider=%s, personality=%s",
        user_id,
        user_email,
        request.provider,
        request.personality,
    )

    # Validate request parameters
    provider = _validate_provider(request.provider)
    personality = _validate_personality(request.personality)

    # Get or create conversation ID
    conversation_id = _get_or_create_conversation_id(request.conversation_id)

    # Get appropriate model for the provider
    model = DEFAULT_MODELS.get(provider, "gpt-4")

    # Check provider API key availability
    provider_key_map = {
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "google": settings.google_api_key,
        "local": "local",  # Local doesn't need API key
    }

    if not provider_key_map.get(provider):
        logger.warning("API key not configured for provider: %s", provider)

        # Try to fallback to another available provider
        fallback_provider = None
        for prov, key in provider_key_map.items():
            if key and prov != "local":
                fallback_provider = prov
                break

        if fallback_provider:
            logger.info("Falling back to provider: %s", fallback_provider)
            provider = fallback_provider
            model = DEFAULT_MODELS.get(provider, "gpt-4")
        else:
            # No providers available, try local
            logger.info("No cloud providers available, attempting local model")
            provider = "local"
            model = DEFAULT_MODELS["local"]

    try:
        # Create assistant service
        assistant_service = _create_ai_assistant_service(settings)

        # Create streaming response
        return StreamingResponse(
            stream_assistant_response(
                assistant_service=assistant_service,
                user_id=str(user_id),
                conversation_id=conversation_id,
                message=request.message,
                provider=provider,
                model=model,
                personality=personality,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "X-Conversation-ID": conversation_id,
                "X-Provider": provider,
                "X-Personality": personality,
            },
        )

    except ProviderInitializationError as e:
        logger.exception("Failed to initialize AI provider")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "provider_unavailable",
                "message": "AI provider is currently unavailable. Please try again later.",
                "provider": provider,
            },
        ) from e

    except Exception as e:
        logger.exception("Unexpected error in ask_assistant endpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_error",
                "message": "An unexpected error occurred. Please try again.",
            },
        ) from e


@router.get(
    "/health",
    summary="AI Assistant Health Check",
    description="Check the health status of the AI assistant service.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "providers_configured": ["openai", "anthropic"],
                        "redis_connected": True,
                        "timestamp": "2024-01-15T10:30:00Z",
                    }
                }
            },
        }
    },
    tags=["assistant"],
)
async def assistant_health_check(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    Check the health status of the AI assistant service.

    Returns information about configured providers and Redis connectivity.
    This endpoint does not require authentication.

    Args:
        settings: Application settings (injected)

    Returns:
        dict: Health status information
    """
    # Check which providers are configured
    configured_providers = []
    if settings.openai_api_key:
        configured_providers.append("openai")
    if settings.anthropic_api_key:
        configured_providers.append("anthropic")
    if settings.google_api_key:
        configured_providers.append("google")

    # Always include local as available
    configured_providers.append("local")

    # Check Redis connectivity
    redis_connected = False
    redis_client = get_redis_client()
    if redis_client:
        try:
            redis_connected = await redis_client.is_connected()
        except Exception:
            redis_connected = False

    return {
        "status": "healthy" if configured_providers else "degraded",
        "providers_configured": configured_providers,
        "default_provider": (
            "openai"
            if "openai" in configured_providers
            else configured_providers[0] if configured_providers else None
        ),
        "redis_connected": redis_connected,
        "supported_personalities": SUPPORTED_PERSONALITIES,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get(
    "/providers",
    summary="List Available AI Providers",
    description="Get a list of supported AI providers and their status.",
    responses={
        200: {
            "description": "List of providers",
            "content": {
                "application/json": {
                    "example": {
                        "providers": [
                            {
                                "name": "openai",
                                "configured": True,
                                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                            }
                        ]
                    }
                }
            },
        }
    },
    tags=["assistant"],
)
async def list_providers(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """
    List all supported AI providers and their configuration status.

    This endpoint does not require authentication and provides information
    about which providers are available for use.

    Args:
        settings: Application settings (injected)

    Returns:
        dict: Provider information including availability and supported models
    """
    providers = [
        {
            "name": "openai",
            "display_name": "OpenAI",
            "configured": bool(settings.openai_api_key),
            "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
            "default_model": "gpt-4",
            "description": "OpenAI GPT models including GPT-4 and GPT-3.5",
        },
        {
            "name": "anthropic",
            "display_name": "Anthropic Claude",
            "configured": bool(settings.anthropic_api_key),
            "models": ["claude-3-5-sonnet-latest", "claude-3-opus", "claude-3-haiku"],
            "default_model": "claude-3-5-sonnet-latest",
            "description": "Anthropic Claude models with strong reasoning capabilities",
        },
        {
            "name": "google",
            "display_name": "Google Gemini",
            "configured": bool(settings.google_api_key),
            "models": ["gemini-2.0-flash", "gemini-pro", "gemini-pro-vision"],
            "default_model": "gemini-2.0-flash",
            "description": "Google Gemini models with multimodal capabilities",
        },
        {
            "name": "local",
            "display_name": "Local Models",
            "configured": True,  # Local is always available if Ollama is running
            "models": ["llama3.2", "mistral", "codellama"],
            "default_model": "llama3.2",
            "description": "Local models via Ollama (requires Ollama installation)",
        },
    ]

    return {
        "providers": providers,
        "default_provider": "openai",
        "supported_personalities": [
            {
                "name": "friendly",
                "description": "Warm, helpful guidance for general platform questions",
            },
            {
                "name": "legal",
                "description": "Serious, professional tone for legal and compensation topics",
            },
        ],
    }
