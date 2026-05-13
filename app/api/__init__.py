"""
META-STAMP V3 API Package.

This package contains all API endpoint implementations for the META-STAMP V3
creator-protection platform. The API is organized by version to support backward
compatibility and future API evolution.

Package Structure:
    - v1/: Version 1 API endpoints (current stable version)
        - auth.py: Authentication endpoints (login, logout, me)
        - upload.py: File upload endpoints (direct, presigned URL, confirmation)
        - fingerprint.py: Asset fingerprinting endpoints
        - assets.py: Asset management endpoints (list, get, delete)
        - wallet.py: Wallet balance and transaction history endpoints
        - analytics.py: AI Touch Value™ calculation endpoints
        - assistant.py: AI assistant chat endpoints
    - v2/: Future API version (planned)

All endpoints are versioned under the /api/v1, /api/v2, etc. URL prefixes
as specified in the Agent Action Plan section 0.3.
"""
