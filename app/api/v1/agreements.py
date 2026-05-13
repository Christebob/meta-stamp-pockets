"""
Agreement and terms endpoints for META-STAMP V3 Pockets.

Public endpoints for viewing terms of service and checking agreement status.
"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.agreement import CURRENT_TERMS_VERSION, TermsResponse


logger = logging.getLogger(__name__)
router = APIRouter(tags=["agreements"])


TERMS_EFFECTIVE_DATE = "2026-04-19"
TERMS_FRESHNESS_WINDOW_DAYS = 180

TERMS_KEY_TERMS = {
    "license_type": "non-exclusive, revocable",
    "price_per_pull_usd": 0.0025,
    "freshness_window_days": TERMS_FRESHNESS_WINDOW_DAYS,
    "creator_compensation": "automatic, per-pull",
    "training_use_allowed": False,
    "attribution_required": True,
    "metering_bypass_prohibited": True,
    "termination": "either party, any time",
    "governing_law": "State of California, USA",
    "patent": "USPTO #63/997,909",
}

TERMS_FULL_TEXT = """
POCKETS CONTENT LICENSE TERMS v1.0.0

Effective Date: 2026-04-19

1. LICENSE GRANT: By connecting to the Pockets MCP server, you (the "Agent Provider")
   are granted a non-exclusive, revocable license to access and retrieve pre-indexed
   content from registered Pockets.

2. METERING: Each content pull is metered and billed to your agent account at the
   per-pull rate set by the content creator (default: $0.0025 USD per pull).

3. CREATOR COMPENSATION: Creators are compensated automatically for each pull of
   their content. You may not circumvent or interfere with the metering system.

4. FRESHNESS WINDOW: Content access obtained via a pull expires after 180 days.
   After expiration, content must be re-pulled and the creator is compensated again.
   This ensures creators receive ongoing compensation as their content remains in use.

5. USAGE RESTRICTIONS: Content retrieved via Pockets may be used for AI model
   responses and outputs. Content may NOT be used for model training without
   separate written agreement with the creator.

6. ATTRIBUTION: When using Pocket content in AI outputs, best-effort attribution
   to the original creator is required.

7. TERMINATION: Either party may terminate this agreement at any time by
   disconnecting from the MCP server or deactivating the API key.

8. GOVERNING LAW: This agreement is governed by the laws of the State of
   California, USA. Any disputes shall be resolved in San Francisco County.

9. PATENT NOTICE: The HTTP 402 micropayment infrastructure underlying this service
   is covered by USPTO patent application #63/997,909.

Acceptance: Connection to this MCP server constitutes acceptance of these terms.
"""


@router.get(
    "/terms",
    response_model=TermsResponse,
    summary="Get current terms",
    description=(
        "Returns the current Pockets Content License Terms as structured JSON. "
        "Includes freshness_window_days (180) — content access expires after this "
        "many days and must be re-pulled."
    ),
)
async def get_terms() -> JSONResponse:
    """Return current terms of service."""
    return JSONResponse(
        content=TermsResponse(
            version=CURRENT_TERMS_VERSION,
            effective_date=TERMS_EFFECTIVE_DATE,
            freshness_window_days=TERMS_FRESHNESS_WINDOW_DAYS,
            summary=(
                "Connection to the Pockets MCP server constitutes acceptance of content "
                "license terms. Each pull is metered at $0.0025 and creators are compensated "
                "automatically. Content access expires after 180 days (freshness window)."
            ),
            key_terms=TERMS_KEY_TERMS,
            full_text=TERMS_FULL_TEXT.strip(),
        ).model_dump(mode="json"),
    )
