"""
LangChain-powered Multi-Provider AI Assistant Service for META-STAMP V3.

This module provides a comprehensive AI assistant service that supports multiple LLM providers
through LangChain's unified interface. The assistant helps creators understand their assets,
fingerprints, AI Touch Values, and legal considerations through conversational interactions.

Supported Providers (per Agent Action Plan section 0.3 - MANDATORY):
- OpenAI GPT-4/5 via langchain-openai
- Anthropic Claude via langchain-anthropic
- Google Gemini via langchain-google-genai
- Local models via standard LangChain interface

Key Features:
- Multi-provider support with runtime switching capability
- Tool calling for fingerprint lookup and analytics queries
- Streaming response support via Server-Sent Events
- Conversation context management in Redis (1-hour TTL)
- Dual personality modes: friendly guidance and serious legal advisory

Per Agent Action Plan:
- Section 0.3: LangChain multi-provider MANDATORY requirements
- Section 0.4: Tool calling and streaming implementation
- Section 0.6: AI assistant service requirements
- Section 0.8: LangChain integration strategies with Redis context

Author: META-STAMP V3 Platform
License: Proprietary
"""

import json
import logging

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

# LangChain model/message imports are deferred to first use to avoid 2-3s startup delay.
# Tool exports are created at module load because tests and callers import them directly.

from app.config import Settings, get_settings
from app.core.database import get_db_client
from app.core.redis_client import RedisClient, get_redis_client
from app.services.ai_value_service import AIValueService
from app.services.fingerprinting_service import FingerprintingService

try:
    from langchain.chat_models import init_chat_model
except Exception as exc:  # pragma: no cover - depends on optional provider packages
    def init_chat_model(*args: Any, **kwargs: Any) -> Any:
        """Placeholder so tests/callers can patch init_chat_model when LangChain is unavailable."""
        raise ProviderInitializationError("LangChain chat model initialization unavailable") from exc


# Configure module logger for tracking assistant operations
logger = logging.getLogger(__name__)

# Constants for conversation management
CONVERSATION_TTL_SECONDS = 3600  # 1-hour TTL per Agent Action Plan section 0.4
CONVERSATION_KEY_PREFIX = "conversation"
MAX_CONTEXT_MESSAGES = 20  # Maximum messages to retain in conversation context
MAX_TOOL_ITERATIONS = 3  # Maximum tool call iterations per message


class AIAssistantError(Exception):
    """Base exception for AI assistant service errors."""


class ProviderInitializationError(AIAssistantError):
    """Raised when AI provider initialization fails."""


class ToolExecutionError(AIAssistantError):
    """Raised when tool execution fails."""


class ConversationContextError(AIAssistantError):
    """Raised when conversation context operations fail."""


class StreamingError(AIAssistantError):
    """Raised when streaming response fails."""


# ===========================================================================
# Tool Functions for AI Assistant
# ===========================================================================

# Global service instances for tool functions (set during service initialization)
_fingerprint_service_instance: FingerprintingService | None = None
_ai_value_service_instance: AIValueService | None = None


async def _lookup_fingerprint_impl(asset_id: str) -> dict[str, Any]:
    """
    Look up fingerprint data for a specific asset.

    This tool retrieves fingerprint information including perceptual hashes,
    spectral analysis data, embeddings, and metadata for a given asset.
    Use this when the user asks about their asset's fingerprint, protection
    status, or unique identification data.

    Args:
        asset_id: The unique identifier of the asset to look up

    Returns:
        Dictionary containing fingerprint data including:
        - fingerprint_id: Unique fingerprint identifier
        - asset_id: Associated asset identifier
        - fingerprint_type: Type of fingerprint (image, audio, video, text)
        - perceptual_hashes: Hash values for content matching (if applicable)
        - spectral_data: Audio spectral analysis (if audio asset)
        - video_hashes: Video frame hashes (if video asset)
        - text_hash: Content hash (if text asset)
        - embeddings: Semantic embeddings for similarity detection
        - processing_status: Current status of fingerprint processing
        - created_at: Timestamp when fingerprint was generated
    """
    logger.info(f"Tool lookup_fingerprint called for asset_id={asset_id}")

    if _fingerprint_service_instance is None:
        logger.error("Fingerprint service not initialized for tool")
        return {
            "error": "Fingerprint service not available",
            "asset_id": asset_id,
            "status": "service_unavailable",
        }

    try:
        # Get database client and query fingerprint
        db_client = get_db_client()
        fingerprint_collection = db_client.get_fingerprints_collection()

        # Query fingerprint by asset_id
        fingerprint_doc = await fingerprint_collection.find_one({"asset_id": asset_id})

        if fingerprint_doc is None:
            logger.info(f"No fingerprint found for asset_id={asset_id}")
            return {
                "asset_id": asset_id,
                "status": "not_found",
                "message": "No fingerprint found for this asset. The asset may still be processing or hasn't been fingerprinted yet.",
            }

        # Format the fingerprint data for assistant response
        result = {
            "fingerprint_id": str(fingerprint_doc.get("_id", "")),
            "asset_id": fingerprint_doc.get("asset_id", asset_id),
            "fingerprint_type": fingerprint_doc.get("fingerprint_type", "unknown"),
            "processing_status": fingerprint_doc.get("processing_status", "unknown"),
            "created_at": (
                fingerprint_doc.get("created_at", "").isoformat()
                if fingerprint_doc.get("created_at")
                else None
            ),
            "status": "found",
        }

        # Include type-specific data
        if fingerprint_doc.get("perceptual_hashes"):
            result["perceptual_hashes"] = {
                "phash": fingerprint_doc["perceptual_hashes"].get("phash", "")[:16] + "...",
                "ahash": fingerprint_doc["perceptual_hashes"].get("ahash", "")[:16] + "...",
                "dhash": fingerprint_doc["perceptual_hashes"].get("dhash", "")[:16] + "...",
            }

        if fingerprint_doc.get("spectral_data"):
            result["spectral_summary"] = {
                "duration": fingerprint_doc["spectral_data"].get("duration"),
                "spectral_centroid_mean": fingerprint_doc["spectral_data"].get(
                    "spectral_centroid_mean"
                ),
            }

        if fingerprint_doc.get("video_hashes"):
            result["video_summary"] = {
                "frames_analyzed": fingerprint_doc["video_hashes"].get("total_frames_analyzed", 0),
                "has_average_hash": bool(fingerprint_doc["video_hashes"].get("average_hash")),
            }

        if fingerprint_doc.get("text_hash"):
            result["text_hash_preview"] = fingerprint_doc["text_hash"][:16] + "..."

        if fingerprint_doc.get("embeddings"):
            result["has_embeddings"] = True
            result["embedding_model"] = fingerprint_doc["embeddings"].get("embedding_model")

        logger.info(f"Fingerprint lookup successful for asset_id={asset_id}")
        return result

    except RuntimeError as e:
        logger.warning(f"Database not initialized for fingerprint lookup: {e}")
        return {
            "asset_id": asset_id,
            "status": "database_unavailable",
            "error": "Database is not currently available",
        }
    except Exception as e:
        logger.exception(f"Error looking up fingerprint for asset_id={asset_id}")
        return {
            "asset_id": asset_id,
            "status": "error",
            "error": str(e),
        }


async def _query_analytics_impl(asset_id: str) -> dict[str, Any]:
    """
    Query AI Touch Value™ analytics for a specific asset.

    This tool retrieves AI Touch Value calculations and analytics data
    for a given asset, including compensation estimates based on the
    formula: AI Touch Value = ModelEarnings * (ContributionScore/100)
    * (ExposureScore/100) * 0.25

    Use this when the user asks about their potential earnings,
    AI Touch Value, compensation estimates, or analytics for their content.

    Args:
        asset_id: The unique identifier of the asset to query analytics for

    Returns:
        Dictionary containing analytics data including:
        - asset_id: The queried asset identifier
        - latest_calculation: Most recent AI Touch Value calculation
        - calculation_history: List of historical calculations
        - formula_breakdown: Step-by-step calculation explanation
        - total_calculated_value: Sum of all calculations for this asset
    """
    logger.info(f"Tool query_analytics called for asset_id={asset_id}")

    if _ai_value_service_instance is None:
        logger.error("AI Value service not initialized for tool")
        return {
            "error": "Analytics service not available",
            "asset_id": asset_id,
            "status": "service_unavailable",
        }

    try:
        # Get calculation history for this asset
        history = await _ai_value_service_instance.get_calculation_history(
            asset_id=asset_id, limit=5
        )

        if not history:
            logger.info(f"No analytics found for asset_id={asset_id}")
            return {
                "asset_id": asset_id,
                "status": "no_data",
                "message": "No AI Touch Value calculations found for this asset. "
                "Calculations are generated when you submit your asset for analysis.",
                "formula_info": {
                    "formula": "AI Touch Value = ModelEarnings * (ContributionScore/100) * (ExposureScore/100) * 0.25",
                    "equity_factor": 0.25,
                    "description": "The 25% equity factor represents the standard creator compensation rate.",
                },
            }

        # Get the latest calculation
        latest = history[0]

        # Calculate total value from history
        total_value = sum(float(calc.get("calculated_value", 0)) for calc in history)

        # Get formula breakdown for the latest calculation
        formula_breakdown = _ai_value_service_instance.get_formula_breakdown(
            model_earnings=float(latest.get("model_earnings", 0)),
            contribution_score=float(latest.get("contribution_score", 0)),
            exposure_score=float(latest.get("exposure_score", 0)),
        )

        result = {
            "asset_id": asset_id,
            "status": "found",
            "latest_calculation": {
                "calculated_value": latest.get("calculated_value"),
                "model_earnings": latest.get("model_earnings"),
                "contribution_score": latest.get("contribution_score"),
                "exposure_score": latest.get("exposure_score"),
                "equity_factor": latest.get("equity_factor", 0.25),
                "timestamp": latest.get("timestamp"),
            },
            "calculation_count": len(history),
            "total_calculated_value": f"{total_value:.2f}",
            "formula_breakdown": formula_breakdown,
            "currency": "USD",
        }

        logger.info(f"Analytics query successful for asset_id={asset_id}")
        return result

    except RuntimeError as e:
        logger.warning(f"Database not initialized for analytics query: {e}")
        return {
            "asset_id": asset_id,
            "status": "database_unavailable",
            "error": "Database is not currently available",
        }
    except Exception as e:
        logger.exception(f"Error querying analytics for asset_id={asset_id}")
        return {
            "asset_id": asset_id,
            "status": "error",
            "error": str(e),
        }


from langchain_core.tools import tool as _lc_tool

lookup_fingerprint = _lc_tool(_lookup_fingerprint_impl)
query_analytics = _lc_tool(_query_analytics_impl)


# ===========================================================================
# AI Assistant Service Class
# ===========================================================================


class AIAssistantService:
    """
    LangChain-powered multi-provider AI assistant service.

    This service provides intelligent conversational assistance to creators using
    the META-STAMP V3 platform. It supports multiple LLM providers (OpenAI, Anthropic,
    Google, local) through LangChain's unified interface and can switch providers
    at runtime.

    Key Capabilities:
    - Multi-provider support: OpenAI GPT-4/5, Anthropic Claude, Google Gemini, local models
    - Tool calling: Lookup fingerprints, query analytics, calculate AI Touch Values
    - Streaming responses: Real-time response streaming via Server-Sent Events
    - Context management: Conversation history stored in Redis with 1-hour TTL
    - Personality modes: Friendly guidance mode and serious legal advisory mode

    Attributes:
        redis: Redis client for conversation context storage
        fingerprint_service: Service for fingerprint lookups (tool calling)
        ai_value_service: Service for analytics queries (tool calling)
        api_keys: Dictionary of provider API keys
        default_provider: Default LLM provider to use
        default_model: Default model name for the provider
        tools: List of tool functions available to the assistant

    Example:
        >>> service = AIAssistantService(
        ...     redis_client=redis,
        ...     fingerprint_service=fp_service,
        ...     ai_value_service=av_service,
        ...     openai_api_key="your-openai-api-key",
        ...     default_provider="openai",
        ...     default_model="gpt-4"
        ... )
        >>> async for chunk in service.send_message_stream(
        ...     user_id="user123",
        ...     conversation_id="conv456",
        ...     message="What is my fingerprint status?"
        ... ):
        ...     print(chunk, end="")
    """

    def __init__(
        self,
        fingerprint_service: FingerprintingService,
        ai_value_service: AIValueService,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        google_api_key: str | None = None,
        default_provider: str = "openai",
        default_model: str = "gpt-4",
    ) -> None:
        """
        Initialize the AIAssistantService with required dependencies.

        Args:
            fingerprint_service: FingerprintingService instance for fingerprint lookups
            ai_value_service: AIValueService instance for analytics queries
            openai_api_key: Optional OpenAI API key for GPT models
            anthropic_api_key: Optional Anthropic API key for Claude models
            google_api_key: Optional Google API key for Gemini models
            default_provider: Default AI provider ("openai", "anthropic", "google", "local")
            default_model: Default model name for the selected provider
        """
        self.fingerprint_service = fingerprint_service
        self.ai_value_service = ai_value_service
        self.logger = logging.getLogger(__name__)

        # Store API keys for provider initialization
        self.api_keys: dict[str, str | None] = {
            "openai": openai_api_key,
            "anthropic": anthropic_api_key,
            "google": google_api_key,
        }

        self.default_provider = default_provider
        self.default_model = default_model

        # Initialize tool functions list
        self.tools = [lookup_fingerprint, query_analytics]

        # Set global service instances for tool functions
        # This is an intentional pattern for dependency injection with module-level tool functions
        global _fingerprint_service_instance, _ai_value_service_instance  # noqa: PLW0603
        _fingerprint_service_instance = fingerprint_service
        _ai_value_service_instance = ai_value_service

        # Validate that at least one provider is configured
        configured_providers = [k for k, v in self.api_keys.items() if v]
        if not configured_providers:
            self.logger.warning(
                "No AI provider API keys configured. "
                "Assistant will operate in limited 'local' mode."
            )
        else:
            self.logger.info(f"AI Assistant initialized with providers: {configured_providers}")

        self.logger.info(
            f"AIAssistantService initialized - default_provider={default_provider}, "
            f"default_model={default_model}"
        )

    def _get_redis_client(self) -> RedisClient | None:
        """Get Redis client with error handling."""
        redis_client = get_redis_client()
        if redis_client is None:
            self.logger.warning("Redis client not available")
        return redis_client

    def _init_model(self, provider: str, model: str) -> Any:
        """
        Initialize a chat model for the specified provider and model.

        Uses LangChain's init_chat_model with provider:model format to create
        a configured model instance with tool calling capabilities.

        Args:
            provider: AI provider name ("openai", "anthropic", "google", "local")
            model: Model name (e.g., "gpt-4", "claude-3-5-sonnet", "gemini-2.0-flash")

        Returns:
            Configured LangChain chat model instance with bound tools

        Raises:
            ProviderInitializationError: If provider initialization fails
        """
        self.logger.info(f"Initializing model: provider={provider}, model={model}")

        try:
            # Build model identifier based on provider
            # LangChain uses provider:model format
            model_kwargs: dict[str, Any] = {}

            if provider == "openai":
                if not self.api_keys.get("openai"):
                    raise ProviderInitializationError(
                        f"OpenAI API key not configured for provider '{provider}'"
                    )
                model_id = f"openai:{model}"
                model_kwargs["api_key"] = self.api_keys["openai"]

            elif provider == "anthropic":
                if not self.api_keys.get("anthropic"):
                    raise ProviderInitializationError(
                        f"Anthropic API key not configured for provider '{provider}'"
                    )
                model_id = f"anthropic:{model}"
                model_kwargs["api_key"] = self.api_keys["anthropic"]

            elif provider == "google":
                if not self.api_keys.get("google"):
                    raise ProviderInitializationError(
                        f"Google API key not configured for provider '{provider}'"
                    )
                # Google uses google_vertexai or google_genai prefix
                model_id = f"google_genai:{model}"
                model_kwargs["api_key"] = self.api_keys["google"]

            elif provider == "local":
                # Local models don't require API keys
                # Use ollama or similar local model format
                model_id = f"ollama:{model}"

            else:
                raise ProviderInitializationError(f"Unsupported provider: {provider}")

            chat_model = init_chat_model(model_id, **model_kwargs)

            # Bind tools to the model for function calling capabilities
            model_with_tools = chat_model.bind_tools(self.tools)

            self.logger.info(f"Model initialized successfully: {model_id}")
            return model_with_tools

        except Exception as e:
            error_msg = f"Failed to initialize model {provider}:{model}: {e}"
            self.logger.exception(error_msg)
            raise ProviderInitializationError(error_msg) from e

    async def _get_conversation_context(
        self, user_id: str, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        Retrieve conversation context from Redis.

        Gets the stored message history for a conversation, which is used
        to maintain context across multiple interactions.

        Args:
            user_id: User identifier
            conversation_id: Unique conversation identifier

        Returns:
            List of message dictionaries with 'role' and 'content' keys.
            Returns empty list if no context exists.
        """
        redis_key = f"{CONVERSATION_KEY_PREFIX}:{user_id}:{conversation_id}"
        self.logger.debug(f"Getting conversation context: {redis_key}")

        redis_client = self._get_redis_client()
        if redis_client is None:
            self.logger.warning("Redis unavailable, returning empty context")
            return []

        try:
            context_json = await redis_client.get_json(redis_key)

            if context_json is None:
                self.logger.debug(f"No existing context found for {redis_key}")
                return []

            if not isinstance(context_json, list):
                self.logger.warning(f"Invalid context format in Redis for {redis_key}")
                return []

            self.logger.debug(f"Retrieved {len(context_json)} messages from context")
            return context_json

        except Exception:
            self.logger.exception("Error retrieving conversation context")
            return []

    async def _save_conversation_context(
        self, user_id: str, conversation_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """
        Save conversation context to Redis with 1-hour TTL.

        Stores the message history for a conversation, enabling context
        persistence across multiple requests. The context expires after
        1 hour per Agent Action Plan section 0.4.

        Args:
            user_id: User identifier
            conversation_id: Unique conversation identifier
            messages: List of message dictionaries to store

        Raises:
            ConversationContextError: If save operation fails
        """
        redis_key = f"{CONVERSATION_KEY_PREFIX}:{user_id}:{conversation_id}"
        self.logger.debug(f"Saving conversation context: {redis_key} ({len(messages)} messages)")

        redis_client = self._get_redis_client()
        if redis_client is None:
            self.logger.warning("Redis unavailable, cannot save context")
            return

        try:
            # Trim to maximum context size
            if len(messages) > MAX_CONTEXT_MESSAGES:
                messages = messages[-MAX_CONTEXT_MESSAGES:]
                self.logger.debug(f"Trimmed context to {MAX_CONTEXT_MESSAGES} messages")

            # Save with 1-hour TTL
            success = await redis_client.set_json(redis_key, messages, ttl=CONVERSATION_TTL_SECONDS)

            if success:
                self.logger.debug(
                    f"Context saved successfully with {CONVERSATION_TTL_SECONDS}s TTL"
                )
            else:
                self.logger.warning(f"Failed to save context to Redis for {redis_key}")

        except Exception as e:
            self.logger.exception("Error saving conversation context")
            raise ConversationContextError(f"Failed to save conversation context: {e}") from e

    def _build_system_prompt(self, personality: str = "friendly") -> str:
        """
        Build the system prompt based on the requested personality mode.

        The assistant supports two personality modes:
        - "friendly": Warm, helpful guidance for general platform questions
        - "legal": Serious, professional tone for legal and compensation topics

        Args:
            personality: Personality mode ("friendly" or "legal")

        Returns:
            System prompt string for the AI assistant
        """
        base_context = """You are an AI assistant for META-STAMP V3, a comprehensive creator-protection platform
that helps creators protect their creative assets and receive fair compensation when their content
is used in AI training.

You have access to the following tools:
1. lookup_fingerprint: Look up fingerprint data for a creator's asset, including perceptual hashes,
   spectral analysis, and embedding data that uniquely identifies their content.
2. query_analytics: Query AI Touch Value analytics for an asset, showing compensation estimates
   based on the formula: AI Touch Value = ModelEarnings * (ContributionScore/100) * (ExposureScore/100) * 0.25

When users ask about their assets, fingerprints, or compensation, use these tools to provide
accurate, up-to-date information."""

        if personality == "friendly":
            personality_context = """

Your personality is warm, friendly, and supportive. You're here to help creators understand:
- How their creative assets are protected
- What fingerprinting means and how it works
- How the AI Touch Value system calculates their potential compensation
- General questions about the META-STAMP platform

Be encouraging and use simple, accessible language. When explaining technical concepts,
use analogies and examples. Celebrate the creator's work and help them feel empowered.

Key points to remember:
- Fingerprints are like digital DNA for creative works
- AI Touch Value uses a 25% equity factor as the standard creator compensation rate
- Perceptual hashes can detect content even if it's been modified slightly
- The platform is designed to protect creators' rights in the AI age"""

        elif personality == "legal":
            personality_context = """

Your personality is professional, serious, and legally-minded. You provide:
- Precise, factual information about asset protection
- Detailed explanations of the AI Touch Value calculation methodology
- Careful, measured responses about legal implications
- Clear disclaimers when appropriate

Important disclaimers to include when relevant:
- You are not a licensed attorney and cannot provide legal advice
- Calculations are estimates based on the platform's methodology
- Creators should consult legal professionals for specific legal matters
- The 25% equity factor is the platform's standard rate, not a legal requirement

Focus on accuracy, transparency, and helping creators understand their options
within the META-STAMP ecosystem. Use precise terminology and provide citations
to specific calculation formulas when discussing compensation."""

        else:
            # Default to friendly if unknown personality
            self.logger.warning(f"Unknown personality '{personality}', defaulting to friendly")
            return self._build_system_prompt("friendly")

        return base_context + personality_context

    def _convert_messages_to_langchain(self, messages: list[dict[str, Any]]) -> list[Any]:
        """
        Convert stored message dictionaries to LangChain message objects.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            List of LangChain BaseMessage objects
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: PLC0415
        langchain_messages: list[Any] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role in {"user", "human"}:
                langchain_messages.append(HumanMessage(content=content))
            elif role in {"assistant", "ai"}:
                langchain_messages.append(AIMessage(content=content))
            elif role == "tool":
                langchain_messages.append(
                    ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", ""))
                )
            else:
                # Default to human message for unknown roles
                langchain_messages.append(HumanMessage(content=content))

        return langchain_messages

    async def _handle_tool_calls(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """
        Execute tool calls and collect results.

        When the AI model invokes tools, this method executes the corresponding
        tool functions and returns the results for inclusion in the conversation.

        Args:
            tool_calls: List of tool call objects from the model response

        Returns:
            List of tool result dictionaries with 'tool_call_id', 'role', and 'content'
        """
        self.logger.info(f"Handling {len(tool_calls)} tool calls")
        results: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = (
                tool_call.get("name", "")
                if isinstance(tool_call, dict)
                else getattr(tool_call, "name", "")
            )
            tool_call_id = (
                tool_call.get("id", "")
                if isinstance(tool_call, dict)
                else getattr(tool_call, "id", "")
            )
            tool_args = (
                tool_call.get("args", {})
                if isinstance(tool_call, dict)
                else getattr(tool_call, "args", {})
            )

            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            try:
                if tool_name == "lookup_fingerprint":
                    result = await lookup_fingerprint.ainvoke(tool_args)
                elif tool_name == "query_analytics":
                    result = await query_analytics.ainvoke(tool_args)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}
                    self.logger.warning(f"Unknown tool requested: {tool_name}")

                results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": json.dumps(result),
                    }
                )

                self.logger.info(f"Tool {tool_name} executed successfully")

            except Exception as e:
                self.logger.exception("Error executing tool %s", tool_name)
                results.append(
                    {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": json.dumps({"error": str(e), "tool": tool_name}),
                    }
                )

        return results

    async def send_message_stream(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        personality: str = "friendly",
    ) -> AsyncIterator[str]:
        """
        Send a message and stream the response.

        This method provides real-time streaming of the AI response, yielding
        chunks as they become available. Ideal for Server-Sent Events (SSE)
        implementation in the API layer.

        Args:
            user_id: User identifier for context retrieval
            conversation_id: Unique conversation identifier
            message: User's message text
            provider: Optional AI provider override (defaults to self.default_provider)
            model: Optional model name override (defaults to self.default_model)
            personality: Personality mode ("friendly" or "legal")

        Yields:
            String chunks of the AI response as they stream in

        Raises:
            ProviderInitializationError: If model initialization fails
            StreamingError: If streaming fails
        """
        self.logger.info(
            f"send_message_stream called: user={user_id}, conv={conversation_id}, "
            f"provider={provider or self.default_provider}"
        )

        # Use defaults if not specified
        provider = provider or self.default_provider
        model = model or self.default_model

        try:
            # Get conversation context
            context = await self._get_conversation_context(user_id, conversation_id)

            # Initialize the chat model
            chat_model = self._init_model(provider, model)

            # Build system prompt
            system_prompt = self._build_system_prompt(personality)

            # Convert context to LangChain messages
            langchain_messages = self._convert_messages_to_langchain(context)

            # Prepend system message and append new user message
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: PLC0415
            messages: list[Any] = [SystemMessage(content=system_prompt)]
            messages.extend(langchain_messages)
            messages.append(HumanMessage(content=message))

            # Track the full response for context saving
            full_response = ""
            iteration_count = 0

            while iteration_count < MAX_TOOL_ITERATIONS:
                iteration_count += 1

                # Stream the response
                async for chunk in chat_model.astream(messages):
                    # Check for tool calls
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        # Handle tool calls
                        tool_results = await self._handle_tool_calls(chunk.tool_calls)

                        # Add AI message with tool calls and tool results to messages
                        ai_message_content = chunk.content if hasattr(chunk, "content") else ""
                        messages.append(
                            AIMessage(content=ai_message_content, tool_calls=chunk.tool_calls)
                        )

                        for tool_result in tool_results:
                            messages.append(
                                ToolMessage(
                                    content=tool_result["content"],
                                    tool_call_id=tool_result["tool_call_id"],
                                )
                            )

                        # Continue the conversation after tool execution
                        break  # Break to re-enter the while loop for next iteration

                    # Regular content chunk
                    if hasattr(chunk, "content") and chunk.content:
                        full_response += chunk.content
                        yield chunk.content

                else:
                    # No tool calls and stream completed normally
                    break

            # Update conversation context
            context.append(
                {"role": "user", "content": message, "timestamp": datetime.now(UTC).isoformat()}
            )
            context.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Save updated context
            await self._save_conversation_context(user_id, conversation_id, context)

            self.logger.info(f"Streaming complete: {len(full_response)} chars")

        except ProviderInitializationError:
            raise
        except Exception as e:
            error_msg = f"Streaming error: {e}"
            self.logger.exception(error_msg)
            raise StreamingError(error_msg) from e

    async def send_message(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        personality: str = "friendly",
    ) -> dict[str, Any]:
        """
        Send a message and get a complete response (non-streaming).

        This method returns the complete response after the AI has finished
        generating it, along with metadata about the interaction.

        Args:
            user_id: User identifier for context retrieval
            conversation_id: Unique conversation identifier
            message: User's message text
            provider: Optional AI provider override
            model: Optional model name override
            personality: Personality mode ("friendly" or "legal")

        Returns:
            Dictionary containing:
            - response: The AI's complete response text
            - conversation_id: The conversation identifier
            - provider: The AI provider used
            - model: The model used
            - personality: The personality mode used
            - timestamp: When the response was generated
            - tool_calls_made: List of tools that were called (if any)

        Raises:
            ProviderInitializationError: If model initialization fails
            AIAssistantError: If message processing fails
        """
        self.logger.info(
            f"send_message called: user={user_id}, conv={conversation_id}, "
            f"provider={provider or self.default_provider}"
        )

        # Use defaults if not specified
        provider = provider or self.default_provider
        model = model or self.default_model

        try:
            # Get conversation context
            context = await self._get_conversation_context(user_id, conversation_id)

            # Initialize the chat model
            chat_model = self._init_model(provider, model)

            # Build system prompt
            system_prompt = self._build_system_prompt(personality)

            # Convert context to LangChain messages
            langchain_messages = self._convert_messages_to_langchain(context)

            # Prepend system message and append new user message
            from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # noqa: PLC0415
            messages: list[Any] = [SystemMessage(content=system_prompt)]
            messages.extend(langchain_messages)
            messages.append(HumanMessage(content=message))

            # Track tool calls for response metadata
            tools_called: list[str] = []
            iteration_count = 0

            while iteration_count < MAX_TOOL_ITERATIONS:
                iteration_count += 1

                # Invoke the model
                response = await chat_model.ainvoke(messages)

                # Check for tool calls
                if hasattr(response, "tool_calls") and response.tool_calls:
                    # Record which tools were called
                    for tc in response.tool_calls:
                        tool_name = (
                            tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                        )
                        tools_called.append(tool_name)

                    # Handle tool calls
                    tool_results = await self._handle_tool_calls(response.tool_calls)

                    # Add AI response with tool calls
                    messages.append(response)

                    # Add tool results
                    for tool_result in tool_results:
                        messages.append(
                            ToolMessage(
                                content=tool_result["content"],
                                tool_call_id=tool_result["tool_call_id"],
                            )
                        )

                    # Continue to get final response
                    continue

                # No tool calls, we have the final response
                break

            # Extract response content
            response_content = response.content if hasattr(response, "content") else str(response)

            # Update conversation context
            timestamp = datetime.now(UTC)
            context.append({"role": "user", "content": message, "timestamp": timestamp.isoformat()})
            context.append(
                {
                    "role": "assistant",
                    "content": response_content,
                    "timestamp": timestamp.isoformat(),
                }
            )

            # Save updated context
            await self._save_conversation_context(user_id, conversation_id, context)

            result = {
                "response": response_content,
                "conversation_id": conversation_id,
                "provider": provider,
                "model": model,
                "personality": personality,
                "timestamp": timestamp.isoformat(),
                "tool_calls_made": tools_called if tools_called else None,
                "message_count": len(context),
            }

            self.logger.info(
                f"Message processed: {len(response_content)} chars, "
                f"tools_called={tools_called if tools_called else 'none'}"
            )

            return result

        except ProviderInitializationError:
            raise
        except Exception as e:
            error_msg = f"Error processing message: {e}"
            self.logger.exception(error_msg)
            raise AIAssistantError(error_msg) from e

    async def clear_conversation(self, user_id: str, conversation_id: str) -> None:
        """
        Clear a conversation's context from Redis.

        Removes all stored message history for the specified conversation,
        effectively starting fresh.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier to clear
        """
        redis_key = f"{CONVERSATION_KEY_PREFIX}:{user_id}:{conversation_id}"
        self.logger.info(f"Clearing conversation: {redis_key}")

        redis_client = self._get_redis_client()
        if redis_client is None:
            self.logger.warning("Redis unavailable, cannot clear conversation")
            return

        try:
            deleted = await redis_client.delete(redis_key)
            if deleted:
                self.logger.info(f"Conversation cleared: {redis_key}")
            else:
                self.logger.debug(f"No conversation to clear: {redis_key}")

        except Exception:
            self.logger.exception("Error clearing conversation")

    async def get_conversation_history(
        self, user_id: str, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        Get the full conversation history.

        Retrieves all stored messages for a conversation, useful for
        displaying chat history in the UI.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier

        Returns:
            List of message dictionaries with role, content, and timestamp
        """
        self.logger.info(f"Getting conversation history: user={user_id}, conv={conversation_id}")

        context = await self._get_conversation_context(user_id, conversation_id)

        self.logger.info(f"Retrieved {len(context)} messages from history")
        return context

    async def switch_provider(
        self, user_id: str, conversation_id: str, new_provider: str, new_model: str
    ) -> dict[str, Any]:
        """
        Switch the AI provider for a conversation.

        Validates that the new provider is available and returns confirmation.
        The conversation context is preserved when switching providers.

        Args:
            user_id: User identifier
            conversation_id: Conversation identifier
            new_provider: New provider name ("openai", "anthropic", "google", "local")
            new_model: New model name for the provider

        Returns:
            Dictionary containing:
            - success: Whether the switch was successful
            - provider: The new provider name
            - model: The new model name
            - message: Confirmation message
            - available_providers: List of configured providers

        Raises:
            ProviderInitializationError: If the new provider is not available
        """
        self.logger.info(
            f"Switching provider: user={user_id}, conv={conversation_id}, "
            f"new_provider={new_provider}, new_model={new_model}"
        )

        # Validate provider availability
        valid_providers = ["openai", "anthropic", "google", "local"]
        if new_provider not in valid_providers:
            raise ProviderInitializationError(
                f"Invalid provider '{new_provider}'. Must be one of: {valid_providers}"
            )

        # Check if API key is available (except for local)
        if new_provider != "local" and not self.api_keys.get(new_provider):
            available = [p for p in ["openai", "anthropic", "google"] if self.api_keys.get(p)]
            available.append("local")  # Local is always available
            raise ProviderInitializationError(
                f"Provider '{new_provider}' is not configured. Available providers: {available}"
            )

        # Test initialization (will raise if invalid)
        try:
            self._init_model(new_provider, new_model)
        except Exception as e:
            raise ProviderInitializationError(
                f"Failed to initialize {new_provider}:{new_model}: {e}"
            ) from e

        available_providers = [p for p in ["openai", "anthropic", "google"] if self.api_keys.get(p)]
        available_providers.append("local")

        result = {
            "success": True,
            "provider": new_provider,
            "model": new_model,
            "message": f"Successfully switched to {new_provider}:{new_model}",
            "available_providers": available_providers,
            "conversation_id": conversation_id,
        }

        self.logger.info(f"Provider switch successful: {new_provider}:{new_model}")
        return result


# ===========================================================================
# Factory Function
# ===========================================================================


def create_ai_assistant_service(
    fingerprint_service: FingerprintingService,
    ai_value_service: AIValueService,
    settings: Settings | None = None,
) -> AIAssistantService:
    """
    Factory function to create an AIAssistantService with settings from config.

    This is the recommended way to create an AIAssistantService instance,
    as it automatically loads configuration from environment variables.

    Args:
        fingerprint_service: FingerprintingService instance for fingerprint lookups
        ai_value_service: AIValueService instance for analytics queries
        settings: Optional Settings instance (loaded from env if not provided)

    Returns:
        Configured AIAssistantService instance

    Example:
        >>> fp_service = FingerprintingService(storage, metadata)
        >>> av_service = AIValueService()
        >>> assistant = create_ai_assistant_service(fp_service, av_service)
    """
    if settings is None:
        settings = get_settings()

    return AIAssistantService(
        fingerprint_service=fingerprint_service,
        ai_value_service=ai_value_service,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        google_api_key=settings.google_api_key,
        default_provider=settings.default_ai_provider,
        default_model=settings.default_ai_model,
    )


# Export all public classes and functions
__all__ = [
    "AIAssistantError",
    "AIAssistantService",
    "ConversationContextError",
    "ProviderInitializationError",
    "StreamingError",
    "ToolExecutionError",
    "create_ai_assistant_service",
    "lookup_fingerprint",
    "query_analytics",
]
