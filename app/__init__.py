"""
META-STAMP V3 Backend Application Package

This package contains the META-STAMP V3 FastAPI application for creator protection
and AI training compensation calculation. The platform provides:

- Multi-modal asset fingerprinting (text, images, audio, video)
- AI training detection (Phase 2)
- AI Touch Value™ compensation calculation
- Hybrid upload architecture with S3/MinIO integration
- Multi-provider AI assistant (OpenAI, Anthropic, Google)
- Authentication with Auth0 and local JWT fallback

Package Structure:
- api/: REST API endpoints organized by version (v1)
- core/: Core infrastructure (database, redis, storage, auth)
- models/: Pydantic data models for all entities
- services/: Business logic layer for all operations
- utils/: Utility functions and helpers
"""

__version__ = "1.0.0"
__app_name__ = "META-STAMP-V3"
