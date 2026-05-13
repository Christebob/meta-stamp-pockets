#!/usr/bin/env python3
"""
META-STAMP V3 FastAPI Application Entry Point

This module serves as the single, canonical entry point for the META-STAMP V3
backend API — a global compensation foundation between AI companies and creators.

It provides:
- FastAPI application initialization with comprehensive configuration
- CORS middleware for secure frontend-to-backend communication
- AI Crawler detection middleware (402 Payment Required → MCP license)
- API router registration under /api/v1 prefix for versioned endpoints
- MCP JSON-RPC 2.0 router for AI agent content access (Pockets)
- Discovery and .well-known/ai-license endpoints for AI agent lookup
- Startup/shutdown lifecycle events for database and cache management
- Health check and readiness endpoints for monitoring
- Request logging middleware for observability

Usage:
    # Run with uvicorn directly
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # Run as Python script
    python main.py
"""

import asyncio
import hashlib
import logging
import os
import time

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import uvicorn

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.v1 import api_router
from app.config import get_settings
from app.core.database import close_db, init_db
from app.core.redis_client import close_redis, init_redis
from app.middleware.ai_crawler_middleware import AICrawlerMiddleware


# =============================================================================
# Logging Configuration
# =============================================================================

# Configure module logger
logger = logging.getLogger(__name__)

# HTTP status code constants
HTTP_ERROR_THRESHOLD = 400  # Status codes >= 400 indicate errors


def configure_logging(log_level: str) -> None:
    """
    Configure application logging with structured format.

    Sets up logging with the specified level and format for both
    the application logger and uvicorn loggers.

    Args:
        log_level: Logging level string (debug, info, warning, error, critical)
    """
    # Map string log level to logging constant
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    level = level_map.get(log_level.lower(), logging.INFO)

    # Configure root logger with structured format
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set level for app loggers
    logging.getLogger("app").setLevel(level)

    # Reduce noise from third-party libraries in production
    if level > logging.DEBUG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("motor").setLevel(logging.WARNING)
        logging.getLogger("redis").setLevel(logging.WARNING)


# =============================================================================
# Application Lifespan Management
# =============================================================================


async def _mongo_reconnect_loop(settings: Any) -> None:
    """
    Persistent background task that retries MongoDB connection every 60 seconds
    until it succeeds. Allows the app to self-heal after a cold-start race condition
    where MongoDB is not yet ready when the backend first starts.
    """
    retry_interval = 30  # seconds
    attempt = 0
    while True:
        attempt += 1
        await asyncio.sleep(retry_interval)
        try:
            logger.info(f"[bg-reconnect] MongoDB reconnect attempt {attempt}...")
            await init_db(settings)
            logger.info("[bg-reconnect] MongoDB connected successfully — self-healed!")
            return  # Success — exit the loop
        except Exception as exc:
            logger.warning(f"[bg-reconnect] MongoDB still not available: {exc}")
            # Increase interval up to 60s max
            retry_interval = min(retry_interval * 2, 60)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manage application lifecycle events for startup and shutdown.

    DB and Redis are initialized in a background task so the app starts
    accepting requests (and passes Railway's healthcheck) immediately.
    The /ready endpoint reports true once the background init completes.
    """
    # Get configuration settings
    settings = get_settings()

    # Configure logging based on settings
    configure_logging(settings.log_level)

    logger.info("=" * 60)
    logger.info("META-STAMP V3 API Starting...")
    logger.info("=" * 60)
    logger.info(f"Application: {settings.app_name}")
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Debug Mode: {settings.debug}")
    logger.info(f"Log Level: {settings.log_level}")

    async def _background_init():
        """Initialize DB and Redis in background so healthcheck passes immediately."""
        # Initialize MongoDB connection — with persistent reconnect on failure
        try:
            logger.info("[bg] Initializing MongoDB connection...")
            await init_db(settings)
            logger.info("[bg] MongoDB connection established")
        except Exception as e:
            logger.error(
                f"[bg] MongoDB init failed: {e} — app running degraded. "
                "Launching persistent reconnect loop..."
            )
            # Launch a persistent background reconnect loop so the app self-heals
            # when MongoDB becomes available (handles Railway cold-start race condition)
            asyncio.create_task(_mongo_reconnect_loop(settings))

        # Initialize Redis connection
        try:
            logger.info("[bg] Initializing Redis connection...")
            await init_redis(settings)
            logger.info("[bg] Redis connection established")
        except Exception as e:
            logger.warning(f"[bg] Redis init failed: {e} — continuing without cache")

        # Seed demo user if SEED_DEMO_USER env var is set
        if os.environ.get("SEED_DEMO_USER", "").lower() in ("1", "true", "yes"):
            try:
                from app.core.database import get_db_client  # noqa: PLC0415
                db_client = get_db_client()
                db = db_client.get_database()
                existing = await db["users"].find_one({"email": "demo@metastamp.io"})
                if not existing:
                    pwd_hash = hashlib.sha256("demo1234".encode()).hexdigest()
                    await db["users"].insert_one({
                        "email": "demo@metastamp.io",
                        "name": "Demo Creator",
                        "role": "creator",
                        "hashed_password": pwd_hash,
                        "auth0_id": None,
                        "avatar_url": None,
                        "bio": "Meta-Stamp demo account for showcasing creator protection features.",
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC),
                        "last_login": None,
                        "is_active": True,
                    })
                    logger.info("[bg] Demo user seeded: demo@metastamp.io / demo1234")
                else:
                    logger.info("[bg] Demo user already exists")
            except Exception as exc:
                logger.warning(f"[bg] Demo user seed failed: {exc}")

        # Always seed the AI agent test user (no env var gate needed)
        # This user has no pockets, so pulling any Dhar Mann pocket triggers 402
        try:
            from app.core.database import get_db_client  # noqa: PLC0415
            db_client = get_db_client()
            db = db_client.get_database()
            existing_agent = await db["users"].find_one({"email": "agent@metastamp.io"})
            if not existing_agent:
                agent_pwd_hash = hashlib.sha256("agent1234".encode()).hexdigest()
                await db["users"].insert_one({
                    "email": "agent@metastamp.io",
                    "name": "AI Agent",
                    "role": "agent",
                    "hashed_password": agent_pwd_hash,
                    "avatar_url": None,
                    "bio": "Test AI agent account for 402 paywall demo.",
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                    "last_login": None,
                    "is_active": True,
                })
                logger.info("[bg] Agent user seeded: agent@metastamp.io / agent1234")
            else:
                logger.info("[bg] Agent user already exists")
        except Exception as exc:
            logger.warning(f"[bg] Agent user seed failed: {exc}")

        logger.info("[bg] Background initialization complete")

    # Launch DB/Redis init in background — app is immediately ready for traffic
    asyncio.create_task(_background_init())

    logger.info("META-STAMP V3 API accepting requests (background init running)")

    # Yield control to the application
    yield

    # Shutdown: cleanup resources
    logger.info("=" * 60)
    logger.info("META-STAMP V3 API Shutting Down...")
    logger.info("=" * 60)

    # Close Redis connection
    try:
        logger.info("Closing Redis connection...")
        await close_redis()
        logger.info("Redis connection closed successfully")
    except Exception:
        logger.exception("Error closing Redis connection")

    # Close MongoDB connection
    try:
        logger.info("Closing MongoDB connection...")
        await close_db()
        logger.info("MongoDB connection closed successfully")
    except Exception:
        logger.exception("Error closing MongoDB connection")

    logger.info("=" * 60)
    logger.info("META-STAMP V3 API Shutdown Complete")
    logger.info("=" * 60)


# =============================================================================
# FastAPI Application Instance
## =============================================================================
# Get settings for initial configuration
# Default APP_ENV to 'production' if not set or invalid — prevents startup crash
import os as _os
if _os.environ.get("APP_ENV", "") not in ("staging", "testing", "production", "development"):
    _os.environ.setdefault("APP_ENV", "production")
try:
    _settings = get_settings()
except Exception:
    _os.environ["APP_ENV"] = "production"
    _settings = get_settings()
# Create FastAPI application with comprehensive metadata
app = FastAPI(
    title="META-STAMP V3 API",
    description=(
        "Global compensation foundation between AI companies and creators. "
        "META-STAMP V3 provides comprehensive asset fingerprinting, AI training detection, "
        "and residual value calculation (AI Touch Value™) to ensure creators are fairly "
        "compensated when their work is used to train AI models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    debug=_settings.debug,
)


# =============================================================================
# Middleware Configuration
# =============================================================================

# AI Crawler Detection Middleware (must be added before other middleware)
# Detects AI crawlers and returns 402 Payment Required with MCP license endpoint
app.add_middleware(AICrawlerMiddleware)

# Configure CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    """
    Middleware for request logging and timing.

    Logs incoming requests with method, path, and measures response time.
    Adds X-Process-Time header to responses for client-side monitoring.
    """
    # Generate request ID for tracing
    request_id = f"{time.time_ns()}"

    # Record start time
    start_time = time.perf_counter()

    # Log incoming request (debug level to avoid noise in production)
    logger.debug(f"Request started: {request.method} {request.url.path} [Request-ID: {request_id}]")

    # Process request
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Request failed: %s %s [Request-ID: %s]",
            request.method,
            request.url.path,
            request_id,
        )
        raise

    # Calculate processing time
    process_time = time.perf_counter() - start_time
    process_time_ms = round(process_time * 1000, 2)

    # Add timing headers
    response.headers["X-Process-Time"] = f"{process_time_ms}ms"
    response.headers["X-Request-ID"] = request_id

    # Log request completion
    log_level = logging.DEBUG if response.status_code < HTTP_ERROR_THRESHOLD else logging.WARNING
    logger.log(
        log_level,
        f"Request completed: {request.method} {request.url.path} "
        f"[Status: {response.status_code}] [Time: {process_time_ms}ms] "
        f"[Request-ID: {request_id}]",
    )

    return response


# =============================================================================
# API Router Registration
# =============================================================================

# Include the v1 API router under /api/v1 prefix
# This aggregates all endpoint routers: auth, upload, fingerprint, assets,
# wallet, analytics, assistant, pockets, agents, agreements, ingestion,
# keymap, youtube, discovery
app.include_router(
    api_router,
    prefix="/api/v1",
)

# Mount MCP server for AI agent content access (Pockets)
try:
    from app.mcp.server import mcp_router  # noqa: PLC0415

    app.include_router(mcp_router)
    logger.info("MCP router mounted at /mcp")
except ImportError as e:
    logger.warning("MCP router not available: %s", e)


# =============================================================================
# Core Endpoints
# =============================================================================


@app.get(
    "/",
    response_class=JSONResponse,
    tags=["root"],
    summary="API Root",
    description="Returns API welcome message and version information",
)
async def root() -> dict[str, Any]:
    """
    Root endpoint returning API welcome message and version.

    Provides basic information about the API including name, version,
    description, and links to documentation.
    """
    return {
        "name": "META-STAMP V3 API",
        "version": "1.0.0",
        "description": (
            "Global compensation foundation between AI companies and creators. "
            "Protecting creator rights through asset fingerprinting and "
            "AI Touch Value™ compensation calculation."
        ),
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "api_prefix": "/api/v1",
        "endpoints": {
            "auth": "/api/v1/auth",
            "upload": "/api/v1/upload",
            "fingerprint": "/api/v1/fingerprint",
            "assets": "/api/v1/assets",
            "wallet": "/api/v1/wallet",
            "analytics": "/api/v1/analytics",
            "assistant": "/api/v1/assistant",
        },
    }


@app.get(
    "/health",
    response_class=JSONResponse,
    tags=["health"],
    summary="Health Check",
    description="Returns health status and current server timestamp for monitoring",
)
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint for monitoring and load balancer integration.

    Returns the current health status of the API server along with
    a UTC timestamp.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0.0",
        "service": "META-STAMP V3 API",
    }


@app.get(
    "/ready",
    response_class=JSONResponse,
    tags=["health"],
    summary="Readiness Check",
    description="Returns readiness status indicating if the service is ready to handle requests",
)
async def readiness_check() -> dict[str, Any]:
    """
    Readiness check endpoint for Kubernetes-style deployments.

    Verifies that the service has completed initialization and is
    ready to handle incoming requests. Unlike the health check,
    this endpoint validates that dependencies (MongoDB, Redis)
    are accessible.
    """
    # Import here to avoid circular imports during initialization
    from app.core.database import get_db_client  # noqa: PLC0415
    from app.core.redis_client import get_redis_client  # noqa: PLC0415

    checks: dict[str, bool] = {}

    # Check MongoDB connection
    try:
        db_client = get_db_client()
        mongodb_healthy = await db_client.ping()
        checks["mongodb"] = mongodb_healthy
    except Exception:
        checks["mongodb"] = False

    # Check Redis connection
    try:
        redis_client = get_redis_client()
        if redis_client:
            redis_healthy = await redis_client.is_connected()
            checks["redis"] = redis_healthy
        else:
            checks["redis"] = False
    except Exception:
        checks["redis"] = False

    # Service is ready if MongoDB is available (Redis is optional)
    is_ready = checks.get("mongodb", False)

    return {
        "ready": is_ready,
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
    }


@app.get("/.well-known/mcp-server-card.json", tags=["discovery"], include_in_schema=False)
async def mcp_server_card() -> dict[str, Any]:
    """Smithery MCP server card for auto-discovery."""
    return {
        "name": "meta-stamp-pockets",
        "displayName": "Meta-Stamp Pockets",
        "description": "The first commercial implementation of HTTP 402 Payment Required for creator content monetization. AI agents pay $0.0025 per content pull. Patent-pending micropayment infrastructure.",
        "version": "1.0.0",
        "transport": "http",
        "endpoint": "https://metastampv3-production.up.railway.app/mcp/",
        "auth": {"type": "bearer"},
        "pricing": {"model": "per-call", "amount": 0.0025, "currency": "USD"}
    }


@app.get("/.well-known/mcp/server-card.json", tags=["discovery"], include_in_schema=False)
async def mcp_nested_server_card() -> dict[str, Any]:
    """MCP server card for directory crawler auto-discovery."""
    return {
        "name": "meta-stamp-pockets",
        "description": (
            "Licensed creator content for AI agents. Tiered per-pull pricing "
            "from $0.001 to $0.25 based on content type."
        ),
        "url": "https://metastampv3-production.up.railway.app/mcp",
        "version": "1.0.0",
        "capabilities": {
            "tools": True,
            "resources": False,
            "prompts": False,
        },
    }


@app.get("/.well-known/ai-license", tags=["discovery"])
async def ai_license_manifest() -> dict[str, Any]:
    """
    Well-known AI license manifest endpoint.

    Returns licensing terms and MCP endpoint information for AI agents
    that follow the Licensed-MCP-Endpoint specification.
    """
    return {
        "version": "1.0",
        "licensing_terms": "https://metastampv3-production.up.railway.app/api/v1/agreements/terms",
        "default_mcp_endpoint": "https://metastampv3-production.up.railway.app/mcp",
        "contact": "https://metastampv3-production.up.railway.app/docs",
        "pricing": "$0.0025 per content pull",
        "patent_filings": "USPTO #63/997,909",
    }


# =============================================================================
# Admin / Demo Seed Endpoint
# =============================================================================


@app.post(
    "/admin/seed-demo",
    response_class=JSONResponse,
    tags=["admin"],
    summary="Seed demo user",
    description="Creates demo@metastamp.io with password demo1234. Protected by SEED_SECRET env var.",
    include_in_schema=False,
)
async def seed_demo_user(request: Request) -> JSONResponse:
    """
    One-shot endpoint to create the demo user when SEED_DEMO_USER seeding
    failed at startup (e.g. MongoDB not yet ready on Railway cold-start).

    Requires the caller to pass the SEED_SECRET env var value as a Bearer
    token so this endpoint cannot be abused by anonymous callers.
    """
    seed_secret = os.environ.get("SEED_SECRET", "")
    auth_header = request.headers.get("Authorization", "")
    provided_token = auth_header.removeprefix("Bearer ").strip()

    if not seed_secret or provided_token != seed_secret:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    try:
        from app.core.database import get_db_client  # noqa: PLC0415
        db_client = get_db_client()
        db = db_client.get_database()
        existing = await db["users"].find_one({"email": "demo@metastamp.io"})
        if existing:
            return JSONResponse(
                status_code=200,
                content={"status": "already_exists", "email": "demo@metastamp.io"},
            )
        pwd_hash = hashlib.sha256("demo1234".encode()).hexdigest()
        await db["users"].insert_one({
            "email": "demo@metastamp.io",
            "name": "Demo Creator",
            "role": "creator",
            "hashed_password": pwd_hash,
            "auth0_id": None,
            "avatar_url": None,
            "bio": "Meta-Stamp demo account for showcasing creator protection features.",
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_login": None,
            "is_active": True,
        })
        logger.info("[seed-demo] Demo user created: demo@metastamp.io")
        return JSONResponse(
            status_code=201,
            content={"status": "created", "email": "demo@metastamp.io", "password": "demo1234"},
        )
    except Exception as exc:
        logger.exception("[seed-demo] Failed to seed demo user")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post(
    "/admin/seed-agent",
    response_class=JSONResponse,
    tags=["admin"],
    summary="Seed AI agent test user",
    description="Creates agent@metastamp.io with password agent1234 for 402 paywall testing.",
    include_in_schema=False,
)
async def seed_agent_user(request: Request) -> JSONResponse:
    """
    Create a second test user (agent@metastamp.io) that does NOT own any
    pockets, so pulling a pocket as this user triggers the 402 paywall.

    Accepts SEED_SECRET env var OR the fallback token 'metastamp-demo-2026'.
    """
    seed_secret = os.environ.get("SEED_SECRET", "")
    # Fallback token for demo environments where SEED_SECRET is not configured
    fallback_token = "metastamp-demo-2026"
    auth_header = request.headers.get("Authorization", "")
    provided_token = auth_header.removeprefix("Bearer ").strip()

    valid = (seed_secret and provided_token == seed_secret) or (provided_token == fallback_token)
    if not valid:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    try:
        from app.core.database import get_db_client  # noqa: PLC0415
        db_client = get_db_client()
        db = db_client.get_database()
        existing = await db["users"].find_one({"email": "agent@metastamp.io"})
        if existing:
            return JSONResponse(
                status_code=200,
                content={"status": "already_exists", "email": "agent@metastamp.io"},
            )
        pwd_hash = hashlib.sha256("agent1234".encode()).hexdigest()
        await db["users"].insert_one({
            "email": "agent@metastamp.io",
            "name": "AI Agent",
            "role": "agent",
            "hashed_password": pwd_hash,
            "avatar_url": None,
            "bio": "Test AI agent account for paywall demo.",
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "last_login": None,
            "is_active": True,
        })
        logger.info("[seed-agent] Agent user created: agent@metastamp.io")
        return JSONResponse(
            status_code=201,
            content={"status": "created", "email": "agent@metastamp.io", "password": "agent1234"},
        )
    except Exception as exc:
        logger.exception("[seed-agent] Failed to seed agent user")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post(
    "/admin/reset-demo-wallet",
    response_class=JSONResponse,
    tags=["admin"],
    summary="Reset demo creator wallet to $0.00",
    description="Zeros out demo@metastamp.io wallet balance for a clean demo run. Uses fallback token.",
    include_in_schema=False,
)
async def reset_demo_wallet(request: Request) -> JSONResponse:
    """
    Reset demo@metastamp.io wallet balance to $0.00 so the before/after
    wallet credit is clean for each demo run.

    Accepts SEED_SECRET env var OR the fallback token 'metastamp-demo-2026' (demo use only).
    """
    seed_secret = os.environ.get("SEED_SECRET", "")
    fallback_token = "metastamp-demo-2026"  # Demo fallback — harmless (resets wallet to $0 only)
    auth_header = request.headers.get("Authorization", "")
    provided_token = auth_header.removeprefix("Bearer ").strip()

    # Accept SEED_SECRET (production) or fallback (demo environments)
    valid = (seed_secret and provided_token == seed_secret) or (provided_token == fallback_token)
    if not valid:
        return JSONResponse(status_code=403, content={"error": "Forbidden"})

    try:
        from app.core.database import get_db_client  # noqa: PLC0415
        db_client = get_db_client()
        db = db_client.get_database()
        # Find the demo user
        demo_user = await db["users"].find_one({"email": "demo@metastamp.io"})
        if not demo_user:
            return JSONResponse(status_code=404, content={"error": "Demo user not found"})
        user_id = str(demo_user["_id"])
        # Reset wallet balance to $0.00
        result = await db["wallet"].update_one(
            {"user_id": user_id},
            {"$set": {
                "balance": 0.0,
                "total_earned": 0.0,
                "pending_earnings": 0.0,
                "updated_at": datetime.now(UTC),
            }},
            upsert=True,
        )
        logger.info("[reset-demo-wallet] Demo wallet reset to $0.00 for user_id=%s", user_id)
        return JSONResponse(
            status_code=200,
            content={
                "status": "reset",
                "email": "demo@metastamp.io",
                "balance": 0.0,
                "matched": result.matched_count,
                "modified": result.modified_count,
            },
        )
    except Exception as exc:
        logger.exception("[reset-demo-wallet] Failed to reset demo wallet")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# =============================================================================
# Demo Wallet Endpoints (public, no auth — demo only)
# =============================================================================


@app.get(
    "/api/v1/demo/wallet",
    response_class=JSONResponse,
    tags=["demo"],
    summary="Demo wallet balance (public)",
    description="Returns the current demo creator wallet balance. No auth required. Demo use only.",
    include_in_schema=False,
)
async def demo_wallet_balance() -> JSONResponse:
    """
    Public demo wallet balance endpoint.

    Returns the live wallet balance for demo@metastamp.io so the pitch deck
    demo can show credits ticking up in real time without requiring auth.
    """
    try:
        from app.core.database import get_db_client  # noqa: PLC0415
        db_client = get_db_client()
        db = db_client.get_database()

        demo_user = await db["users"].find_one({"email": "demo@metastamp.io"})
        if not demo_user:
            return JSONResponse(
                status_code=404,
                content={"error": "Demo user not found — run /admin/seed-demo first"},
            )

        user_id = str(demo_user["_id"])
        wallet = await db["wallet"].find_one({"user_id": user_id})

        balance = float(wallet.get("balance", 0.0) or 0.0) if wallet else 0.0
        total_earned = float(wallet.get("total_earned", 0.0) or 0.0) if wallet else 0.0
        updated_at = wallet.get("updated_at", datetime.now(UTC)).isoformat() if wallet else datetime.now(UTC).isoformat()

        return JSONResponse(content={
            "demo": True,
            "email": "demo@metastamp.io",
            "balance": balance,
            "total_earned": total_earned,
            "currency": "USD",
            "last_updated": updated_at,
        })
    except Exception as exc:
        logger.exception("[demo-wallet] Failed to read demo wallet")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post(
    "/api/v1/demo/reset",
    response_class=JSONResponse,
    tags=["demo"],
    summary="Reset demo wallet to $1.00 (public)",
    description="Resets the demo creator wallet to $1.00 for a clean demo run. No auth required. Demo use only.",
    include_in_schema=False,
)
async def demo_wallet_reset() -> JSONResponse:
    """
    Public demo wallet reset endpoint.

    Sets demo@metastamp.io wallet balance to $1.00 so each demo run starts
    from a consistent, non-zero state that clearly shows earnings accruing.
    No auth required — this is intentionally public for demo use.
    """
    try:
        from app.core.database import get_db_client  # noqa: PLC0415
        db_client = get_db_client()
        db = db_client.get_database()

        demo_user = await db["users"].find_one({"email": "demo@metastamp.io"})
        if not demo_user:
            return JSONResponse(
                status_code=404,
                content={"error": "Demo user not found — run /admin/seed-demo first"},
            )

        user_id = str(demo_user["_id"])
        now = datetime.now(UTC)
        result = await db["wallet"].update_one(
            {"user_id": user_id},
            {"$set": {
                "balance": 1.0,
                "total_earned": 1.0,
                "pending_earnings": 0.0,
                "updated_at": now,
            }},
            upsert=True,
        )
        logger.info("[demo-reset] Demo wallet reset to $1.00 for user_id=%s", user_id)
        return JSONResponse(content={
            "demo": True,
            "status": "reset",
            "email": "demo@metastamp.io",
            "balance": 1.0,
            "currency": "USD",
            "matched": result.matched_count,
            "modified": result.modified_count,
        })
    except Exception as exc:
        logger.exception("[demo-reset] Failed to reset demo wallet")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# =============================================================================
# Demo Route
# =============================================================================


@app.get("/demo", include_in_schema=False)
@app.get("/demo.html", include_in_schema=False)
async def serve_demo() -> HTMLResponse:
    """Serve the Pockets interactive demo page."""
    import pathlib  # noqa: PLC0415
    demo_path = pathlib.Path(__file__).parent / "demo" / "pockets-demo.html"
    if demo_path.exists():
        return HTMLResponse(content=demo_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Demo not found</h1>", status_code=404)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(404)
async def not_found_handler(request: Request, _exc: Exception) -> JSONResponse:
    """
    Custom 404 Not Found exception handler.

    Returns a consistent JSON response for 404 errors with
    helpful information about the requested path.
    """
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"The requested path '{request.url.path}' was not found",
            "status_code": 404,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Custom 500 Internal Server Error exception handler.

    Logs the error and returns a generic error message to avoid
    exposing internal details to clients.
    """
    logger.error(
        f"Internal server error on {request.method} {request.url.path}: {exc!s}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
        },
    )


# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    """
    Run the application directly using uvicorn.

    This block allows running the API server by executing:
        python main.py

    Configuration is loaded from Settings and applied to uvicorn.
    For production deployments, use:
        uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    settings = get_settings()

    # Log startup configuration
    print("=" * 60)
    print("META-STAMP V3 API Server Starting...")
    print("=" * 60)
    print(f"Host: {settings.host}")
    print(f"Port: {settings.port}")
    print(f"Debug/Reload: {settings.debug}")
    print(f"Log Level: {settings.log_level}")
    print("=" * 60)

    # Run uvicorn with settings from configuration
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level,
        access_log=settings.debug,
    )
