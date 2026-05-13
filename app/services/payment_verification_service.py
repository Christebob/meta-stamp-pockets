"""
Payment Verification Service for META-STAMP V3.

Verifies Stripe PaymentIntent tokens before granting pocket content access.
Never raises — always returns bool to avoid blocking content delivery on errors.
All verification attempts are logged for audit trail.
"""

import logging

import stripe

from app.config import get_settings

logger = logging.getLogger(__name__)


class PaymentVerificationService:
    """
    Verifies Stripe PaymentIntent IDs as proof of payment before content delivery.

    Design principles:
    - Never raises: all failures return False and log the reason
    - Strict: status, amount, and optional pocket_id metadata are all checked
    - Audit-friendly: every verification attempt is logged with outcome
    """

    def __init__(self) -> None:
        settings = get_settings()
        stripe.api_key = settings.stripe_secret_key

    async def verify_payment_token(
        self,
        token: str,
        expected_amount: float,
        pocket_id: str,
    ) -> bool:
        """
        Verify a Stripe PaymentIntent ID as proof of payment.

        Checks (all must pass):
          1. Token format: must start with 'pi_'
          2. PaymentIntent exists in Stripe and is retrievable
          3. status == 'succeeded'
          4. amount (in cents) >= expected_amount * 100
          5. If metadata.pocket_id is set, it must match the requested pocket_id

        Args:
            token: Stripe PaymentIntent ID (expected prefix: 'pi_')
            expected_amount: Required payment amount in USD (e.g., 0.0025)
            pocket_id: Pocket being accessed — validated against intent metadata if present

        Returns:
            True if all checks pass. False on any failure (never raises).
        """
        if not token or not token.startswith("pi_"):
            logger.warning(
                "Payment verification rejected — token does not start with 'pi_': prefix=%s pocket=%s",
                token[:8] if token else "empty",
                pocket_id,
            )
            return False

        try:
            intent = stripe.PaymentIntent.retrieve(token)
        except stripe.error.AuthenticationError:
            logger.error(
                "Stripe authentication failed — check STRIPE_SECRET_KEY configuration (pocket=%s)",
                pocket_id,
            )
            return False
        except stripe.error.InvalidRequestError as exc:
            logger.warning(
                "Invalid Stripe PaymentIntent ID '%s' for pocket=%s: %s",
                token,
                pocket_id,
                exc,
            )
            return False
        except stripe.error.StripeError as exc:
            logger.error(
                "Stripe API error verifying token '%s' for pocket=%s: %s",
                token,
                pocket_id,
                exc,
            )
            return False
        except Exception:
            logger.exception(
                "Unexpected error verifying Stripe token '%s' for pocket=%s",
                token,
                pocket_id,
            )
            return False

        # ── Check 1: status must be 'succeeded' ──────────────────────────────
        intent_status = intent.get("status")
        if intent_status != "succeeded":
            logger.warning(
                "PaymentIntent '%s' rejected — status='%s' (not 'succeeded') pocket=%s",
                token,
                intent_status,
                pocket_id,
            )
            return False

        # ── Check 2: amount must cover expected price (amounts in cents) ──────
        expected_cents = int(round(expected_amount * 100))
        actual_cents = intent.get("amount", 0)
        if actual_cents < expected_cents:
            logger.warning(
                "PaymentIntent '%s' rejected — amount=%d cents < required=%d cents pocket=%s",
                token,
                actual_cents,
                expected_cents,
                pocket_id,
            )
            return False

        # ── Check 3: metadata.pocket_id must match if set ─────────────────────
        metadata = intent.get("metadata") or {}
        meta_pocket_id = metadata.get("pocket_id")
        if meta_pocket_id and meta_pocket_id != pocket_id:
            logger.warning(
                "PaymentIntent '%s' rejected — metadata.pocket_id='%s' != requested='%s'",
                token,
                meta_pocket_id,
                pocket_id,
            )
            return False

        logger.info(
            "PaymentIntent '%s' VERIFIED for pocket=%s (amount=%d cents, status=succeeded)",
            token,
            pocket_id,
            actual_cents,
        )
        return True
