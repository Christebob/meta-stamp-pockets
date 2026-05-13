"""
Core infrastructure services for the META-STAMP V3 backend application.

This package contains the foundational infrastructure components that provide:
- auth: Authentication and authorization services (Auth0 + local JWT fallback)
- database: MongoDB async client with Motor driver and connection pooling
- redis_client: Redis async client for caching and session management
- storage: S3-compatible storage client for MinIO/AWS S3 operations

All services in this package are designed for async operation and follow
the singleton pattern for efficient resource management.
"""
