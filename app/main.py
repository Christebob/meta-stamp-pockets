"""
META-STAMP V3 API - FastAPI Application Entry Point.

This module serves as the comprehensive entry point for the META-STAMP V3 backend API.
It initializes the FastAPI application with CORS middleware, registers all API v1
routers, and configures startup/shutdown event handlers for MongoDB and Redis
connections.

META-STAMP V3 is a global compensation foundation between AI companies and creators,
providing:
- Multi-modal asset fingerprinting (images, audio, video, text)
- AI training detection (Phase 2)
- AI Touch Value™ calculation for creator compensation
- Hybrid upload architecture with S3 presigned URLs
- Multi-provider AI assistant via LangChain

Architecture Decisions:
- FastAPI framework for high-performance async API (MANDATORY per Agent Action Plan 0.3)
- CORS middleware configured for frontend access at configured origins
- Startup/shutdown event handlers for graceful connection management
- Health check endpoint for container orchestration monitoring
- All API endpoints versioned under /api/v1 prefix

Based on Agent Action Plan sections:
- 0.3: FastAPI framework requirement (MANDATORY)
- 0.4: Backend Architecture Implementation (CORS, router registration, events)
- 0.6: main.py transformation mapping
- 0.10: Execution parameters and success criteria
"""

import asyncio
from datetime import UTC, datetime

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.config import Settings
from app.core.database import close_db, init_db
from app.core.redis_client import close_redis, init_redis
from app.mcp.server import mcp_router


# =============================================================================
# Configuration Loading
# =============================================================================

# Load application configuration from environment variables
# Settings uses Pydantic Settings for validation and type safety
settings = Settings()


# =============================================================================
# FastAPI Application Initialization
# =============================================================================

app = FastAPI(
    title="META-STAMP V3 API",
    version="1.0.0",
    description="Global compensation foundation between AI companies and creators",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# CORS Middleware Configuration
# =============================================================================

# Configure Cross-Origin Resource Sharing (CORS) to enable frontend communication.
# The frontend React application runs on a different origin (typically localhost:3000
# in development) and requires CORS headers to make API requests.
#
# Configuration per Agent Action Plan section 0.4:
# - allow_origins: List from settings.cors_origins (defaults to ["http://localhost:3000"])
# - allow_credentials: True for cookie/auth header support
# - allow_methods: All HTTP methods for REST API operations
# - allow_headers: All headers for authorization and content-type
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Startup Event Handler
# =============================================================================


async def _background_startup() -> None:
    """Run heavy startup tasks in background so /health is immediately available."""
    import os
    try:
        await init_db()
    except Exception as exc:
        print(f"⚠️ Warning: Failed to initialize database connection: {exc}")

    try:
        await init_redis()
    except Exception as exc:
        print(f"⚠️ Warning: Failed to initialize Redis connection: {exc}")

    if os.environ.get("SEED_DEMO_USER", "").lower() in ("1", "true", "yes"):
        try:
            from app.core.database import get_db_client
            import hashlib
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
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "last_login": None,
                    "is_active": True,
                })
                print("✅ Demo user seeded: demo@metastamp.io / demo1234")
            # Ensure wallet exists
            user = await db["users"].find_one({"email": "demo@metastamp.io"})
            if user:
                user_id = str(user["_id"])
                w_existing = await db["wallets"].find_one({"user_id": user_id})
                if not w_existing:
                    await db["wallets"].insert_one({
                        "user_id": user_id,
                        "balance": 0.21,
                        "currency": "USD",
                        "pending_earnings": 0.0,
                        "total_earned": 0.21,
                        "total_paid_out": 0.0,
                        "last_payout_at": None,
                        "created_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow(),
                    })
                    print("✅ Demo wallet seeded")
        except Exception as exc:
            print(f"⚠️ Demo seed failed: {exc}")

    print(f"✅ META-STAMP V3 background startup complete")


@app.on_event("startup")
async def startup_event() -> None:
    """
    Handle application startup tasks.

    This event handler runs when the FastAPI application starts. It initializes
    critical services required for META-STAMP V3 operation:

    1. MongoDB Database Connection:
       - Establishes async connection using Motor driver
       - Configures connection pooling (10-100 connections per settings)
       - Creates database indexes for optimized query performance
       - Verifies connectivity with ping command

    2. Redis Cache Connection:
       - Establishes async connection for caching and session management
       - Configures connection with retry logic
       - Verifies connectivity with ping command

    The startup sequence is critical for application functionality. If database
    or Redis connections fail, the application will log errors but continue
    starting to allow health check endpoints to report the issue.

    Per Agent Action Plan section 0.4:
    - Must implement startup events for MongoDB and Redis connections
    - Must log successful startup with host:port information
    """
    # Fire heavy startup tasks in background so /health is immediately available
    asyncio.create_task(_background_startup())
    print("✅ META-STAMP V3 startup — background init scheduled")




# =============================================================================
# Shutdown Event Handler
# =============================================================================


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Handle application shutdown tasks.

    This event handler runs when the FastAPI application is shutting down.
    It ensures graceful cleanup of all external connections:

    1. MongoDB Database Connection:
       - Closes all connections in the pool
       - Releases database resources
       - Ensures no orphaned connections remain

    2. Redis Cache Connection:
       - Flushes pending cache operations
       - Closes connection pool
       - Releases Redis resources

    Graceful shutdown is critical for:
    - Preventing connection leaks
    - Ensuring pending operations complete
    - Maintaining clean state for container restarts

    Per Agent Action Plan section 0.4:
    - Must implement shutdown events for connection cleanup
    - Must log shutdown completion status
    """
    try:
        # Close MongoDB database connection and release pool resources
        await close_db()
    except Exception as exc:
        print(f"⚠️ Warning: Error closing database connection: {exc}")

    try:
        # Close Redis cache connection and flush pending operations
        await close_redis()
    except Exception as exc:
        print(f"⚠️ Warning: Error closing Redis connection: {exc}")

    # Log successful shutdown message per spec
    print("✅ META-STAMP V3 API shutdown complete")


# =============================================================================
# Root Endpoint
# =============================================================================


@app.get("/", tags=["root"])
async def root() -> dict:
    """
    Root endpoint providing API information and navigation.

    This endpoint serves as the API entry point, providing essential information
    about the META-STAMP V3 platform and links to documentation.

    Returns:
        dict: API metadata containing:
            - name: API display name
            - version: Current API version
            - description: Brief platform description
            - docs: Path to interactive Swagger documentation

    Example Response:
        {
            "name": "META-STAMP V3 API",
            "version": "1.0.0",
            "description": "Global compensation foundation between AI companies and creators",
            "docs": "/docs"
        }
    """
    return {
        "name": "META-STAMP V3 API",
        "version": "1.0.0",
        "description": "Global compensation foundation between AI companies and creators",
        "docs": "/docs",
    }


# =============================================================================
# Health Check Endpoint
# =============================================================================


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """
    Health check endpoint for container orchestration and monitoring.

    This endpoint provides basic health status for the META-STAMP V3 API service.
    It is designed for use with container orchestration systems (Docker, Kubernetes)
    to determine if the service is running and responsive.

    The endpoint returns immediately without checking backend dependencies,
    providing a fast liveness probe. For dependency health checks, use the
    /ready endpoint (if implemented) or individual service health endpoints.

    Returns:
        dict: Health status containing:
            - status: Health status string ("healthy")
            - timestamp: ISO 8601 formatted UTC timestamp
            - service: Service identifier for monitoring systems

    Example Response:
        {
            "status": "healthy",
            "timestamp": "2025-01-15T10:30:00.000000",
            "service": "META-STAMP V3 Backend"
        }

    Usage:
        - Docker HEALTHCHECK: curl http://localhost:8000/health
        - Kubernetes livenessProbe: GET /health
        - Load balancer health check: GET /health

    Per Agent Action Plan section 0.10:
    - Must include health check endpoint for container orchestration
    - Must return healthy status with timestamp and service identifier
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "META-STAMP V3 Backend",
    }


# =============================================================================
# API Router Registration
# =============================================================================

# Register the aggregated v1 API router containing all META-STAMP V3 endpoints.
#
# Router Structure (per Agent Action Plan section 0.4):
# - /api/v1/auth: Authentication endpoints (login, logout, me)
# - /api/v1/upload: File upload endpoints (direct, presigned URL, confirmation)
# - /api/v1/fingerprint: Asset fingerprinting endpoints
# - /api/v1/assets: Asset management endpoints (list, get, delete)
# - /api/v1/wallet: Wallet balance and transaction history
# - /api/v1/analytics: AI Touch Value™ calculation endpoints
# - /api/v1/assistant: AI assistant chat endpoints with streaming
#
# All endpoints are versioned under /api/v1 prefix per Agent Action Plan section 0.3
# to support future API versions (/api/v2, /api/v3) without breaking changes.
app.include_router(api_router, prefix="/api/v1")

# Mount MCP server for AI agent content access (Pockets)
app.include_router(mcp_router)

# Mount Discovery router for AI agent content lookup
try:
    from app.api.v1.discovery import router as discovery_router
    app.include_router(discovery_router, prefix="/api/v1/discovery")
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning("Discovery router not available: %s", e)


@app.get("/.well-known/mcp/server-card.json", tags=["discovery"], include_in_schema=False)
async def mcp_nested_server_card() -> dict:
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
async def ai_license_manifest() -> dict:
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
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # Run the application using Uvicorn ASGI server.
    #
    # Configuration from Settings:
    # - host: Server bind address (default: 0.0.0.0 for container access)
    # - port: Server listen port (default: 8000)
    # - reload: Enable hot-reload in debug mode for development
    # - log_level: Uvicorn logging level (debug, info, warning, error)
    #
    # Production Considerations:
    # - Use gunicorn with uvicorn workers for production deployment
    # - Configure multiple workers based on CPU cores
    # - Disable reload in production for stability
    #
    # Example production command:
    #   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level,
    )
