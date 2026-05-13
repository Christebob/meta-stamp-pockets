"""
Stripe Connect API routes for META-STAMP V3 creator payouts.

Endpoints:
- POST /stripe/connect/onboard     Start Stripe Connect onboarding for a creator
- GET  /stripe/connect/status      Check onboarding/payout readiness
- GET  /stripe/connect/complete    Return URL after Stripe onboarding completes
- POST /stripe/connect/payout      Trigger a manual payout to creator
- GET  /stripe/connect/balance     Get creator Stripe Connect balance
- GET  /stripe/config              Public Stripe config (publishable key, fees)
"""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.core.auth import get_current_user
from app.core.database import get_db_client
from app.services import stripe_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stripe"])

STRIPE_ACCOUNTS_COLLECTION = "stripe_connect_accounts"


# =============================================================================
# Request / Response Models
# =============================================================================


class OnboardRequest(BaseModel):
    refresh_url: str = Field(
        default="https://meta-stamp-v3.onrender.com/dashboard",
        description="URL if the creator needs to restart onboarding",
    )
    return_url: str = Field(
        default="https://meta-stamp-v3.onrender.com/dashboard?stripe=complete",
        description="URL after onboarding completes",
    )


class OnboardResponse(BaseModel):
    onboarding_url: str
    account_id: str


class AccountStatusResponse(BaseModel):
    connected: bool
    account_id: str | None = None
    charges_enabled: bool = False
    payouts_enabled: bool = False
    details_submitted: bool = False
    requirements: dict[str, Any] = Field(default_factory=dict)


class PayoutRequest(BaseModel):
    amount_usd: float = Field(..., gt=0, description="Amount in USD to transfer")
    description: str = Field(
        default="Meta-Stamp creator payout",
        description="Description for the transfer",
    )


class PayoutResponse(BaseModel):
    transfer_id: str
    amount_usd: float
    currency: str
    destination: str
    created: int


class StripeConfigResponse(BaseModel):
    publishable_key: str
    platform_fee_percent: float
    minimum_payout_usd: float
    payout_enabled: bool


class BalanceResponse(BaseModel):
    account_id: str
    available: list[dict[str, Any]]
    pending: list[dict[str, Any]]


# =============================================================================
# Helper: get or store Stripe account mapping in MongoDB
# =============================================================================


async def _get_stripe_account_id(user_id: str) -> str | None:
    """Look up the Stripe Connect account ID for a user from MongoDB."""
    try:
        db = get_db_client()
        collection = db.get_database()[STRIPE_ACCOUNTS_COLLECTION]
        doc = await collection.find_one({"user_id": user_id})
        if doc:
            return doc.get("stripe_account_id")
    except RuntimeError:
        logger.warning("Database not available for Stripe account lookup")
    return None


async def _save_stripe_account_id(user_id: str, account_id: str) -> None:
    """Store the Stripe Connect account mapping in MongoDB."""
    try:
        db = get_db_client()
        collection = db.get_database()[STRIPE_ACCOUNTS_COLLECTION]
        await collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "stripe_account_id": account_id,
                    "user_id": user_id,
                },
                "$setOnInsert": {"created_at": datetime.now(UTC)},
            },
            upsert=True,
        )
    except RuntimeError:
        logger.error("Database not available - cannot save Stripe account mapping")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        )


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/config",
    response_model=StripeConfigResponse,
    summary="Get public Stripe configuration",
)
async def get_stripe_config() -> StripeConfigResponse:
    """Return public Stripe config (publishable key, fee structure)."""
    settings = get_settings()
    return StripeConfigResponse(
        publishable_key=settings.stripe_publishable_key or "",
        platform_fee_percent=settings.stripe_platform_fee_percent,
        minimum_payout_usd=settings.stripe_minimum_payout_cents / 100.0,
        payout_enabled=settings.stripe_payout_enabled,
    )


@router.get(
    "/connect/onboard",
    response_model=OnboardResponse,
    summary="Get Stripe Connect onboarding URL",
)
async def get_onboarding_url(
    refresh_url: str = "https://meta-stamp-v3.onrender.com/dashboard",
    return_url: str = "https://meta-stamp-v3.onrender.com/dashboard?stripe=complete",
    current_user: dict[str, Any] = Depends(get_current_user),
) -> OnboardResponse:
    """
    Return a Stripe Connect Express onboarding URL for the authenticated creator.

    If the creator already has a Connect account, generates a fresh link.
    If not, creates a new Express account first.
    """
    user_id = str(current_user.get("_id", ""))
    email = current_user.get("email", "")
    name = current_user.get("full_name") or current_user.get("username") or email

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    existing_account_id = await _get_stripe_account_id(user_id)

    if existing_account_id:
        try:
            url = await stripe_service.create_account_link(
                existing_account_id, refresh_url, return_url
            )
            return OnboardResponse(onboarding_url=url, account_id=existing_account_id)
        except Exception as e:
            logger.error("Failed to create account link for %s: %s", existing_account_id, e)
            raise HTTPException(status_code=502, detail="Stripe account link creation failed")

    try:
        account_id = await stripe_service.create_connect_account(email, name)
        await _save_stripe_account_id(user_id, account_id)
        url = await stripe_service.create_account_link(account_id, refresh_url, return_url)
        return OnboardResponse(onboarding_url=url, account_id=account_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Stripe onboarding (GET) failed for user %s: %s", user_id, e)
        raise HTTPException(status_code=502, detail="Stripe onboarding failed")


@router.get(
    "/connect/callback",
    summary="Handle Stripe Connect OAuth callback",
)
async def connect_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Handle the OAuth callback redirect from Stripe after Express onboarding.

    For Express accounts, Stripe redirects the creator back to `return_url`
    after onboarding completes. This endpoint checks the resulting account
    status and returns it to the caller.

    If Stripe sends an `error` param, returns the error details without raising
    so the frontend can display a user-friendly message.
    """
    user_id = str(current_user.get("_id", ""))

    if error:
        logger.warning(
            "Stripe Connect callback error for user %s: %s — %s",
            user_id,
            error,
            error_description or "",
        )
        return {
            "status": "error",
            "error": error,
            "message": error_description or "Stripe onboarding was not completed",
        }

    account_id = await _get_stripe_account_id(user_id)
    if not account_id:
        return {
            "status": "no_account",
            "message": "No Stripe Connect account found — complete onboarding first",
        }

    try:
        status_data = await stripe_service.get_account_status(account_id)
        return {
            "status": "complete" if status_data["details_submitted"] else "incomplete",
            "account_id": account_id,
            "charges_enabled": status_data["charges_enabled"],
            "payouts_enabled": status_data["payouts_enabled"],
            "requirements": status_data.get("requirements", {}),
        }
    except Exception as e:
        logger.error("Callback status check failed for account %s: %s", account_id, e)
        return {"status": "error", "message": "Could not verify onboarding status"}


@router.post(
    "/connect/onboard",
    response_model=OnboardResponse,
    summary="Start Stripe Connect onboarding",
)
async def start_onboarding(
    request: OnboardRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> OnboardResponse:
    """
    Create a Stripe Connect Express account and return the onboarding URL.

    If the creator already has an account, generates a fresh onboarding link.
    """
    user_id = str(current_user.get("_id", ""))
    email = current_user.get("email", "")
    name = current_user.get("full_name") or current_user.get("username") or email

    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid user")

    existing_account_id = await _get_stripe_account_id(user_id)

    if existing_account_id:
        try:
            url = await stripe_service.create_account_link(
                existing_account_id, request.refresh_url, request.return_url
            )
            return OnboardResponse(onboarding_url=url, account_id=existing_account_id)
        except Exception as e:
            logger.error("Failed to create account link: %s", e)
            raise HTTPException(status_code=502, detail="Stripe account link creation failed")

    try:
        account_id = await stripe_service.create_connect_account(email, name)
        await _save_stripe_account_id(user_id, account_id)
        url = await stripe_service.create_account_link(
            account_id, request.refresh_url, request.return_url
        )
        return OnboardResponse(onboarding_url=url, account_id=account_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Stripe onboarding failed: %s", e)
        raise HTTPException(status_code=502, detail="Stripe onboarding failed")


@router.get(
    "/connect/status",
    response_model=AccountStatusResponse,
    summary="Check Stripe Connect status",
)
async def get_connect_status(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> AccountStatusResponse:
    """Check whether the creator has connected Stripe and payout readiness."""
    user_id = str(current_user.get("_id", ""))
    account_id = await _get_stripe_account_id(user_id)

    if not account_id:
        return AccountStatusResponse(connected=False)

    try:
        status_data = await stripe_service.get_account_status(account_id)
        return AccountStatusResponse(
            connected=True,
            account_id=account_id,
            charges_enabled=status_data["charges_enabled"],
            payouts_enabled=status_data["payouts_enabled"],
            details_submitted=status_data["details_submitted"],
            requirements=status_data.get("requirements", {}),
        )
    except Exception as e:
        logger.error("Failed to get Stripe status for %s: %s", account_id, e)
        return AccountStatusResponse(connected=True, account_id=account_id)


@router.get(
    "/connect/complete",
    summary="Stripe onboarding completion redirect",
)
async def onboarding_complete(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Called after creator finishes Stripe onboarding. Returns updated status."""
    user_id = str(current_user.get("_id", ""))
    account_id = await _get_stripe_account_id(user_id)

    if not account_id:
        return {"status": "no_account", "message": "No Stripe account found"}

    try:
        status_data = await stripe_service.get_account_status(account_id)
        return {
            "status": "complete" if status_data["details_submitted"] else "incomplete",
            "account_id": account_id,
            "payouts_enabled": status_data["payouts_enabled"],
        }
    except Exception as e:
        logger.error("Error checking onboarding completion: %s", e)
        return {"status": "error", "message": "Could not verify onboarding status"}


@router.post(
    "/connect/payout",
    response_model=PayoutResponse,
    summary="Trigger a payout to creator",
)
async def create_payout(
    request: PayoutRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> PayoutResponse:
    """Transfer funds from the platform to the creator's Stripe Connect account."""
    user_id = str(current_user.get("_id", ""))
    account_id = await _get_stripe_account_id(user_id)

    if not account_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Stripe Connect account found. Complete onboarding first.",
        )

    settings = get_settings()
    if not settings.stripe_payout_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Payouts are currently disabled",
        )

    min_payout = settings.stripe_minimum_payout_cents / 100.0
    if request.amount_usd < min_payout:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Minimum payout is ${min_payout:.2f}",
        )

    try:
        result = await stripe_service.create_payout(
            account_id, request.amount_usd, request.description
        )
        return PayoutResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Payout failed for %s: %s", account_id, e)
        raise HTTPException(status_code=502, detail="Stripe payout failed")


@router.get(
    "/connect/balance",
    response_model=BalanceResponse,
    summary="Get creator Stripe balance",
)
async def get_creator_balance(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BalanceResponse:
    """Return the available and pending balance for the creator's Connect account."""
    user_id = str(current_user.get("_id", ""))
    account_id = await _get_stripe_account_id(user_id)

    if not account_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No Stripe Connect account found. Complete onboarding first.",
        )

    try:
        result = await stripe_service.get_balance(account_id)
        return BalanceResponse(**result)
    except Exception as e:
        logger.error("Failed to get balance for %s: %s", account_id, e)
        raise HTTPException(status_code=502, detail="Could not retrieve Stripe balance")
