"""
Pockets API endpoints for META-STAMP V3.

Endpoints:
- POST /pockets: Create a creator pocket from URL content
- GET /pockets: List authenticated creator pockets
- POST /pockets/{pocket_id}/pull: Pull content with 402 paywall enforcement
- POST /pockets/{pocket_id}/create-payment-intent: Create Stripe PaymentIntent for a pull
"""

import asyncio
import logging
import time

from datetime import UTC, datetime
from typing import Any

import stripe

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.core.auth import get_current_user
from app.core.database import get_db_client
from app.models.pocket import (
    PocketCreateRequest,
    PocketPullResponse,
    PocketResponse,
    PocketStatus,
)
from app.services.metering_service import MeteringService
from app.services.payment_verification_service import PaymentVerificationService
from app.services.pocket_service import (
    PocketNotFoundError,
    PocketService,
    PocketStateError,
    PocketValidationError,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["pockets"])

DEFAULT_PRICE_PER_PULL = 0.0025

# Sentinel token accepted only when DEMO_MODE=true
_DEMO_TOKEN = "demo-payment-settled"


def get_pocket_service() -> PocketService:
    """Dependency provider for PocketService."""
    return PocketService()


def _resolve_creator_id(current_user: dict[str, Any]) -> str:
    """Resolve authenticated creator ID from current user context."""
    creator_id = current_user.get("_id") or current_user.get("id")
    if not isinstance(creator_id, str) or not creator_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authenticated user context",
        )
    return creator_id.strip()


@router.post(
    "/",
    response_model=PocketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create pocket",
    description="Create a new pocket by indexing content from a submitted URL.",
)
async def create_pocket(
    request: PocketCreateRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
    pocket_service: PocketService = Depends(get_pocket_service),
) -> JSONResponse:
    """Create a pocket and attempt snapshot extraction immediately."""
    creator_id = _resolve_creator_id(current_user)

    try:
        pocket = await pocket_service.create_pocket(
            creator_id=creator_id,
            content_url=str(request.content_url),
        )
        response = PocketResponse.from_pocket(pocket)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=response.model_dump(mode="json"),
        )

    except PocketValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError:
        logger.exception("Database unavailable during pocket creation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from None
    except Exception:
        logger.exception("Unexpected error creating pocket")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while creating pocket",
        ) from None


@router.get(
    "/",
    response_model=list[PocketResponse],
    status_code=status.HTTP_200_OK,
    summary="List pockets",
    description="List pockets for the authenticated creator.",
)
async def list_pockets(
    limit: int = Query(default=50, ge=1, le=200),
    current_user: dict[str, Any] = Depends(get_current_user),
    pocket_service: PocketService = Depends(get_pocket_service),
) -> JSONResponse:
    """List creator-owned pockets in reverse chronological order."""
    creator_id = _resolve_creator_id(current_user)

    try:
        pockets = await pocket_service.list_pockets(creator_id=creator_id, limit=limit)
        response_payload = [
            PocketResponse.from_pocket(pocket).model_dump(mode="json") for pocket in pockets
        ]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_payload,
        )

    except PocketValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError:
        logger.exception("Database unavailable while listing pockets")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from None
    except Exception:
        logger.exception("Unexpected error listing pockets")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while listing pockets",
        ) from None


# ─────────────────────────────────────────────────────────────────────────────
# Internal wallet helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _get_wallet_balance(user_id: str) -> float:
    """Read the current wallet balance for a user. Returns 0.0 if no wallet."""
    try:
        db_client = get_db_client()
        wallet_col = db_client.get_wallet_collection()
        wallet = await wallet_col.find_one({"user_id": user_id})
        if wallet:
            return float(wallet.get("balance", 0.0))
        return 0.0
    except Exception:
        logger.exception("Failed to read wallet balance for user %s", user_id)
        return 0.0


async def _deduct_wallet(user_id: str, amount: float) -> bool:
    """
    Atomically deduct `amount` from the user's wallet.

    Uses a conditional update (balance >= amount) to prevent overdraft.
    Returns True if deduction succeeded, False otherwise.
    """
    try:
        db_client = get_db_client()
        wallet_col = db_client.get_wallet_collection()
        now = datetime.now(UTC)

        result = await wallet_col.update_one(
            {"user_id": user_id, "balance": {"$gte": amount}},
            {
                "$inc": {"balance": -round(amount, 6)},
                "$set": {"updated_at": now},
            },
        )
        return result.modified_count == 1
    except Exception:
        logger.exception("Failed to deduct wallet for user %s", user_id)
        return False


async def _credit_owner_wallet(owner_id: str, amount: float, pocket_id: str) -> None:
    """
    Credit the pocket owner's wallet and log the pull via MeteringService.

    Fire-and-forget — failures are logged but never block the response.
    """
    try:
        metering = MeteringService()
        await metering.log_pull(
            pocket_id=pocket_id,
            agent_key_id="rest_api",
            creator_id=owner_id,
            provider="rest",
            compensation_amount=amount,
            response_time_ms=0.0,
        )
    except Exception:
        logger.exception(
            "Failed to credit owner %s for pocket %s (non-blocking)",
            owner_id,
            pocket_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Paywall-enforced pull endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/{pocket_id}/pull",
    response_model=PocketPullResponse,
    status_code=status.HTTP_200_OK,
    summary="Pull pocket content (paywall enforced)",
    description=(
        "Pull content from a pocket. Requires authentication and payment. "
        "X-Payment-Token must be either a real Stripe PaymentIntent ID (pi_...) "
        "or 'demo-payment-settled' when DEMO_MODE is enabled. "
        "Returns 402 if payment is absent or invalid."
    ),
    responses={
        402: {
            "description": "Payment required — invalid or missing payment token / insufficient balance",
            "content": {
                "application/json": {
                    "example": {
                        "error": "payment_required",
                        "price": 0.0025,
                        "pocket_id": "abc123",
                        "message": "Payment required to access this content",
                    }
                }
            },
        },
    },
)
async def pull_pocket(
    pocket_id: str,
    request: Request,
    current_user: dict[str, Any] = Depends(get_current_user),
    pocket_service: PocketService = Depends(get_pocket_service),
    x_payment_token: str | None = Header(default=None, alias="X-Payment-Token"),
) -> JSONResponse:
    """
    Pull content from a pocket with paywall enforcement.

    Payment token resolution (for non-owners):
      - pi_... token → verified via Stripe PaymentIntent (real payment)
      - 'demo-payment-settled' → accepted only when DEMO_MODE=true
      - any other value → rejected with HTTP 402
      - no token → wallet balance checked and deducted

    All payment attempts (real and demo) are logged for audit trail.
    """
    settings = get_settings()
    caller_id = _resolve_creator_id(current_user)
    start_time = time.monotonic()

    # ── 1. Look up pocket ────────────────────────────────────────────────────
    try:
        pocket = await pocket_service.get_pocket_by_id(pocket_id)
    except PocketValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except PocketNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if pocket.status != PocketStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only active pockets can be pulled",
        )

    # ── 2. Determine price ───────────────────────────────────────────────────
    price = pocket.price_per_pull if pocket.price_per_pull > 0 else DEFAULT_PRICE_PER_PULL

    # ── 3. Owner gets free access ────────────────────────────────────────────
    is_owner = (caller_id == pocket.creator_id)

    # ── 4–5. Payment resolution for non-owners ───────────────────────────────
    payment_settled = False
    payment_method: str = "owner"  # audit label

    if not is_owner:
        if x_payment_token:
            token = x_payment_token.strip()

            if token.startswith("pi_"):
                # ── Real Stripe PaymentIntent verification ────────────────────
                verifier = PaymentVerificationService()
                verified = await verifier.verify_payment_token(
                    token=token,
                    expected_amount=price,
                    pocket_id=pocket_id,
                )
                if not verified:
                    logger.warning(
                        "Stripe payment verification FAILED for pocket=%s user=%s token=%s",
                        pocket_id,
                        caller_id,
                        token[:16],
                    )
                    return JSONResponse(
                        status_code=402,
                        content={
                            "error": "payment_required",
                            "price": price,
                            "pocket_id": pocket_id,
                            "currency": "USD",
                            "message": "Stripe PaymentIntent verification failed",
                            "accepted_methods": [
                                "stripe_payment_intent",
                                "wallet_balance",
                            ],
                        },
                    )
                payment_settled = True
                payment_method = "stripe_payment_intent"
                logger.info(
                    "PAYMENT[stripe] verified pocket=%s user=%s token=%s",
                    pocket_id,
                    caller_id,
                    token[:16],
                )

            elif token == _DEMO_TOKEN:
                # ── Demo / pilot mode token ───────────────────────────────────
                if not settings.demo_mode:
                    logger.warning(
                        "Demo token rejected — DEMO_MODE=false pocket=%s user=%s",
                        pocket_id,
                        caller_id,
                    )
                    return JSONResponse(
                        status_code=402,
                        content={
                            "error": "payment_required",
                            "price": price,
                            "pocket_id": pocket_id,
                            "currency": "USD",
                            "message": (
                                "Demo tokens are not accepted in production. "
                                "Provide a valid Stripe PaymentIntent ID (pi_...)."
                            ),
                            "accepted_methods": ["stripe_payment_intent", "wallet_balance"],
                        },
                    )
                payment_settled = True
                payment_method = "demo"
                logger.info(
                    "PAYMENT[demo] accepted pocket=%s user=%s (DEMO_MODE=true)",
                    pocket_id,
                    caller_id,
                )

            else:
                # ── Unknown / unrecognised token — reject ─────────────────────
                logger.warning(
                    "Payment token rejected — unrecognised format pocket=%s user=%s prefix=%s",
                    pocket_id,
                    caller_id,
                    token[:8],
                )
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "payment_required",
                        "price": price,
                        "pocket_id": pocket_id,
                        "currency": "USD",
                        "message": (
                            "Invalid X-Payment-Token. Provide a Stripe PaymentIntent ID (pi_...) "
                            "obtained from POST /pockets/{pocket_id}/create-payment-intent."
                        ),
                        "accepted_methods": ["stripe_payment_intent", "wallet_balance"],
                    },
                )

        else:
            # ── No token — fall back to wallet balance ────────────────────────
            balance = await _get_wallet_balance(caller_id)
            if balance < price:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "payment_required",
                        "price": price,
                        "pocket_id": pocket_id,
                        "currency": "USD",
                        "wallet_balance": round(balance, 6),
                        "message": "Payment required to access this content",
                        "accepted_methods": [
                            "wallet_balance",
                            "stripe_payment_intent",
                        ],
                    },
                )
            deducted = await _deduct_wallet(caller_id, price)
            if not deducted:
                return JSONResponse(
                    status_code=402,
                    content={
                        "error": "payment_required",
                        "price": price,
                        "pocket_id": pocket_id,
                        "currency": "USD",
                        "message": "Wallet deduction failed — insufficient balance",
                        "accepted_methods": [
                            "wallet_balance",
                            "stripe_payment_intent",
                        ],
                    },
                )
            payment_settled = True
            payment_method = "wallet"
            logger.info(
                "PAYMENT[wallet] deducted $%.6f pocket=%s user=%s",
                price,
                pocket_id,
                caller_id,
            )

    # ── 6. Execute pull ──────────────────────────────────────────────────────
    try:
        updated_pocket = await pocket_service.pull_pocket(
            creator_id=pocket.creator_id,
            pocket_id=pocket_id,
        )
    except PocketStateError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Unexpected error during pocket pull")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while pulling pocket",
        ) from None

    elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

    # Credit owner wallet (fire-and-forget for non-owner pulls)
    if not is_owner:
        asyncio.create_task(
            _credit_owner_wallet(
                owner_id=pocket.creator_id,
                amount=price,
                pocket_id=pocket_id,
            )
        )

    logger.info(
        "PULL[success] pocket=%s user=%s method=%s elapsed_ms=%.2f",
        pocket_id,
        caller_id,
        payment_method,
        elapsed_ms,
    )

    response = PocketPullResponse(
        pocket=PocketResponse.from_pocket(updated_pocket),
        retrieved_content=updated_pocket.snapshot_text or "",
        compensation_increment=price,
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.model_dump(mode="json"),
        headers={"X-Response-Time-Ms": str(elapsed_ms)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Create-payment-intent endpoint (legitimate Stripe flow)
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/{pocket_id}/create-payment-intent",
    status_code=status.HTTP_201_CREATED,
    summary="Create Stripe PaymentIntent for a pocket pull",
    description=(
        "Creates a Stripe PaymentIntent for the price of a single pocket pull. "
        "AI Operators call this, complete payment on the client, then use the "
        "returned payment_intent_id as the X-Payment-Token header when calling pull."
    ),
)
async def create_payment_intent(
    pocket_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
    pocket_service: PocketService = Depends(get_pocket_service),
) -> JSONResponse:
    """
    Legitimate payment flow:
      1. Client calls this endpoint → receives client_secret + payment_intent_id
      2. Client completes payment via Stripe.js / SDK using client_secret
      3. Client calls POST /pockets/{pocket_id}/pull with X-Payment-Token: <payment_intent_id>
    """
    settings = get_settings()
    caller_id = _resolve_creator_id(current_user)

    # ── Look up pocket ───────────────────────────────────────────────────────
    try:
        pocket = await pocket_service.get_pocket_by_id(pocket_id)
    except PocketValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except PocketNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    if pocket.status != PocketStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only active pockets can be purchased",
        )

    price = pocket.price_per_pull if pocket.price_per_pull > 0 else DEFAULT_PRICE_PER_PULL
    amount_cents = max(1, int(round(price * 100)))  # Stripe minimum is 1 cent

    # ── Create Stripe PaymentIntent ──────────────────────────────────────────
    stripe.api_key = settings.stripe_secret_key

    try:
        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="usd",
            metadata={
                "pocket_id": pocket_id,
                "caller_id": caller_id,
                "platform": "meta-stamp-v3",
            },
            description=f"Meta-Stamp pocket pull: {pocket_id}",
        )
    except stripe.error.AuthenticationError:
        logger.error("Stripe authentication failed creating PaymentIntent for pocket=%s", pocket_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Payment service temporarily unavailable",
        ) from None
    except stripe.error.StripeError as exc:
        logger.error(
            "Stripe error creating PaymentIntent for pocket=%s: %s",
            pocket_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to create payment intent",
        ) from None
    except Exception:
        logger.exception("Unexpected error creating PaymentIntent for pocket=%s", pocket_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while creating payment intent",
        ) from None

    logger.info(
        "PaymentIntent created: id=%s pocket=%s user=%s amount_cents=%d",
        intent.id,
        pocket_id,
        caller_id,
        amount_cents,
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "payment_intent_id": intent.id,
            "client_secret": intent.client_secret,
            "amount": price,
            "amount_cents": amount_cents,
            "currency": "usd",
            "pocket_id": pocket_id,
            "instructions": (
                "Complete payment using client_secret via Stripe.js, "
                "then call POST /pockets/{pocket_id}/pull with "
                "X-Payment-Token: <payment_intent_id>"
            ),
        },
    )
