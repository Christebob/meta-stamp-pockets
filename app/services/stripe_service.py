"""
Stripe Connect service for Meta-Stamp creator payouts.

Handles creating Connect accounts, onboarding links, transfers, and balance queries.
All operations use Stripe test mode when keys begin with sk_test_.
"""

import logging
from typing import Any

import stripe

from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_stripe() -> None:
    """Configure Stripe API key from settings."""
    settings = get_settings()
    if not settings.stripe_secret_key:
        raise ValueError("STRIPE_SECRET_KEY is not configured")
    stripe.api_key = settings.stripe_secret_key


async def create_connect_account(creator_email: str, creator_name: str) -> str:
    """
    Create a Stripe Connect Express account for a creator.

    Returns the Stripe account ID (acct_...).
    """
    _get_stripe()
    try:
        account = stripe.Account.create(
            type="express",
            email=creator_email,
            capabilities={
                "transfers": {"requested": True},
            },
            business_profile={
                "name": creator_name,
                "product_description": "Creator content licensing via Meta-Stamp",
            },
            metadata={"platform": "meta-stamp-v3"},
        )
        logger.info("Created Stripe Connect account %s for %s", account.id, creator_email)
        return account.id
    except stripe.StripeError as e:
        logger.error("Failed to create Stripe Connect account: %s", e)
        raise


async def create_account_link(account_id: str, refresh_url: str, return_url: str) -> str:
    """
    Generate an onboarding URL for a Connect account.

    Returns the URL the creator should be redirected to complete onboarding.
    """
    _get_stripe()
    try:
        link = stripe.AccountLink.create(
            account=account_id,
            refresh_url=refresh_url,
            return_url=return_url,
            type="account_onboarding",
        )
        return link.url
    except stripe.StripeError as e:
        logger.error("Failed to create account link for %s: %s", account_id, e)
        raise


async def get_account_status(account_id: str) -> dict[str, Any]:
    """
    Return onboarding and capability status for a Connect account.
    """
    _get_stripe()
    try:
        account = stripe.Account.retrieve(account_id)
        return {
            "account_id": account_id,
            "charges_enabled": account.charges_enabled,
            "payouts_enabled": account.payouts_enabled,
            "details_submitted": account.details_submitted,
            "requirements": {
                "currently_due": account.requirements.currently_due or [],
                "eventually_due": account.requirements.eventually_due or [],
            },
        }
    except stripe.StripeError as e:
        logger.error("Failed to retrieve account status for %s: %s", account_id, e)
        raise


async def create_payout(
    account_id: str,
    amount_usd: float,
    description: str,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """
    Transfer funds from the platform to a creator's Connect account.

    amount_usd is converted to cents for the Stripe API.
    idempotency_key prevents duplicate transfers on retry — callers should
    supply a stable, deterministic key tied to the triggering event.
    Returns the Transfer object details.
    """
    _get_stripe()
    amount_cents = int(round(amount_usd * 100))
    if amount_cents <= 0:
        raise ValueError(f"Payout amount must be positive, got {amount_usd}")
    try:
        create_kwargs: dict[str, Any] = {
            "amount": amount_cents,
            "currency": "usd",
            "destination": account_id,
            "description": description,
            "metadata": {"platform": "meta-stamp-v3"},
        }
        if idempotency_key:
            create_kwargs["idempotency_key"] = idempotency_key

        transfer = stripe.Transfer.create(**create_kwargs)
        logger.info(
            "Transferred $%.4f to account %s (transfer %s, idempotency_key=%s)",
            amount_usd,
            account_id,
            transfer.id,
            idempotency_key or "none",
        )
        return {
            "transfer_id": transfer.id,
            "amount_usd": amount_usd,
            "amount_cents": amount_cents,
            "currency": transfer.currency,
            "destination": transfer.destination,
            "created": transfer.created,
        }
    except stripe.StripeError as e:
        logger.error("Failed to create transfer for %s: %s", account_id, e)
        raise


async def get_balance(account_id: str) -> dict[str, Any]:
    """
    Return the available and pending balance for a Connect account.
    """
    _get_stripe()
    try:
        balance = stripe.Balance.retrieve(stripe_account=account_id)
        available = [
            {"amount": b.amount / 100, "currency": b.currency}
            for b in balance.available
        ]
        pending = [
            {"amount": b.amount / 100, "currency": b.currency}
            for b in balance.pending
        ]
        return {
            "account_id": account_id,
            "available": available,
            "pending": pending,
        }
    except stripe.StripeError as e:
        logger.error("Failed to retrieve balance for %s: %s", account_id, e)
        raise
