"""
Metering service for META-STAMP V3 Pockets per-pull micro-payments.

Tracks every content pull, calculates compensation, credits creator wallets,
and triggers Stripe Connect payouts to creator bank accounts.

Pull logging is async (fire-and-forget) to keep the hot path fast.
"""

import hashlib
import logging
import time

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId

from app.core.database import get_db_client


logger = logging.getLogger(__name__)

_STRIPE_ACCOUNTS_COLLECTION = "stripe_connect_accounts"


class MeteringService:
    """Service for tracking pulls and managing creator compensation."""

    def __init__(self) -> None:
        self._stripe_service = None

    def _get_stripe_service(self):
        """
        Lazy-load the Stripe service module.

        Returns the stripe_service module if Stripe is configured and payouts
        are enabled, otherwise returns None so callers skip payout cleanly.
        """
        if self._stripe_service is None:
            try:
                from app.config import get_settings
                settings = get_settings()
                if not settings.stripe_secret_key:
                    logger.debug("STRIPE_SECRET_KEY not set — payouts disabled")
                    return None
                if not settings.stripe_payout_enabled:
                    logger.debug("stripe_payout_enabled=False — payouts disabled")
                    return None
                from app.services import stripe_service as _stripe_mod
                self._stripe_service = _stripe_mod
                logger.info("Stripe Connect service loaded (test_mode=%s)", settings.stripe_secret_key.startswith("sk_test_"))
            except ImportError:
                logger.warning("stripe library not installed — payouts disabled")
                return None
        return self._stripe_service

    async def log_pull(
        self,
        pocket_id: str,
        agent_key_id: str,
        creator_id: str,
        provider: str,
        compensation_amount: float,
        response_time_ms: float = 0.0,
    ) -> None:
        """
        Log a content pull, credit the creator's wallet, and trigger Stripe payout.

        This method is designed to be called fire-and-forget from the MCP
        hot path. It should not raise exceptions that would block the response.
        """
        try:
            db_client = get_db_client()

            # 1. Insert pull log
            pull_logs = db_client.get_pull_logs_collection()
            now = datetime.now(UTC)
            pull_doc: dict[str, Any] = {
                "pocket_id": pocket_id,
                "agent_key_id": agent_key_id,
                "creator_id": creator_id,
                "provider": provider,
                "compensation_amount": round(compensation_amount, 6),
                "response_time_ms": round(response_time_ms, 2),
                "pulled_at": now,
                "metadata": {},
            }
            await pull_logs.insert_one(pull_doc)

            # 2. Increment pocket pull count and compensation
            if ObjectId.is_valid(pocket_id):
                pockets = db_client.get_pockets_collection()
                await pockets.update_one(
                    {"_id": ObjectId(pocket_id)},
                    {
                        "$inc": {
                            "pull_count": 1,
                            "compensation_earned": round(compensation_amount, 6),
                        },
                        "$set": {"updated_at": now},
                    },
                )

            # 3. Credit creator wallet (upsert)
            wallet = db_client.get_wallet_collection()
            await wallet.update_one(
                {"user_id": creator_id},
                {
                    "$inc": {
                        "balance": round(compensation_amount, 6),
                        "total_earned": round(compensation_amount, 6),
                    },
                    "$set": {"updated_at": now},
                    "$setOnInsert": {
                        "user_id": creator_id,
                        "currency": "USD",
                        "pending_earnings": 0.0,
                        "total_paid_out": 0.0,
                        "created_at": now,
                    },
                },
                upsert=True,
            )

            logger.debug(
                "Pull logged: pocket=%s, agent=%s, amount=%.6f, time=%.2fms",
                pocket_id,
                agent_key_id,
                compensation_amount,
                response_time_ms,
            )

            # 4. Trigger Stripe payout (fire-and-forget, never blocks)
            await self._trigger_stripe_payout(
                creator_id=creator_id,
                amount_usd=compensation_amount,
                pocket_id=pocket_id,
            )

        except Exception:
            # Never let metering failures block the hot path
            logger.exception("Failed to log pull for pocket %s", pocket_id)

    async def _trigger_stripe_payout(
        self,
        creator_id: str,
        amount_usd: float,
        pocket_id: str,
    ) -> None:
        """
        Attempt to trigger a Stripe transfer to the creator's Connect account.

        This is fire-and-forget — any failure is logged but never propagated.
        The creator's wallet balance is already credited before this runs;
        funds remain in the wallet even if Stripe is unreachable.

        Split: creator receives (100 - platform_fee_percent)% of the pull
        compensation, minimum threshold enforced before issuing any transfer.
        """
        try:
            stripe_svc = self._get_stripe_service()
            if stripe_svc is None:
                return

            from app.config import get_settings
            settings = get_settings()

            # Apply platform split — creator gets (100 - fee)%
            creator_share = 1.0 - (settings.stripe_platform_fee_percent / 100.0)
            creator_amount_usd = round(amount_usd * creator_share, 6)

            # Enforce minimum payout threshold
            min_payout_usd = settings.stripe_minimum_payout_cents / 100.0
            if creator_amount_usd < min_payout_usd:
                logger.debug(
                    "Skipping payout: $%.6f below minimum $%.2f for creator %s",
                    creator_amount_usd,
                    min_payout_usd,
                    creator_id,
                )
                return

            # Look up creator's connected Stripe account
            db_client = get_db_client()
            accounts_col = db_client.get_database()[_STRIPE_ACCOUNTS_COLLECTION]
            doc = await accounts_col.find_one({"user_id": creator_id})
            if not doc or not doc.get("stripe_account_id"):
                logger.debug("No Stripe Connect account for creator %s — skipping payout", creator_id)
                return

            stripe_account_id = doc["stripe_account_id"]

            # Deterministic idempotency key — deduplicates retries within a
            # 60-second window for the same creator/pocket/amount triple.
            minute_window = int(time.time() // 60)
            amount_cents = int(round(creator_amount_usd * 100))
            key_data = f"ms-payout:{creator_id}:{pocket_id}:{amount_cents}:{minute_window}"
            idempotency_key = "ms-" + hashlib.sha256(key_data.encode()).hexdigest()[:40]

            result = await stripe_svc.create_payout(
                account_id=stripe_account_id,
                amount_usd=creator_amount_usd,
                description=f"MCP pull compensation — Pocket {pocket_id}",
                idempotency_key=idempotency_key,
            )

            logger.info(
                "Stripe transfer complete: %s — $%.6f (%.0f%% of $%.6f) to creator %s",
                result.get("transfer_id"),
                creator_amount_usd,
                (1.0 - creator_share) * 100,  # fee percent shown in log
                amount_usd,
                creator_id,
            )
        except Exception:
            # Never let Stripe failures block the hot path
            logger.exception("Stripe payout failed for creator %s (non-blocking)", creator_id)

    async def get_pull_stats(self, pocket_id: str) -> dict[str, Any]:
        """Get pull statistics for a pocket."""
        db_client = get_db_client()
        pull_logs = db_client.get_pull_logs_collection()

        pipeline = [
            {"$match": {"pocket_id": pocket_id}},
            {
                "$group": {
                    "_id": "$pocket_id",
                    "total_pulls": {"$sum": 1},
                    "total_compensation": {"$sum": "$compensation_amount"},
                    "avg_response_time_ms": {"$avg": "$response_time_ms"},
                    "unique_agents": {"$addToSet": "$agent_key_id"},
                    "first_pull": {"$min": "$pulled_at"},
                    "last_pull": {"$max": "$pulled_at"},
                }
            },
        ]

        results = await pull_logs.aggregate(pipeline).to_list(length=1)
        if not results:
            return {
                "total_pulls": 0,
                "total_compensation": 0.0,
                "avg_response_time_ms": 0.0,
                "unique_agents": 0,
            }

        stats = results[0]
        average_response_time = stats.get("avg_response_time_ms", 0.0)
        return {
            "total_pulls": stats.get("total_pulls", 0),
            "total_compensation": round(stats.get("total_compensation", 0.0), 6),
            "avg_response_time_ms": round(average_response_time if average_response_time else 0.0, 2),
            "unique_agents": len(stats.get("unique_agents", [])),
            "first_pull": stats["first_pull"].isoformat() if stats.get("first_pull") else None,
            "last_pull": stats["last_pull"].isoformat() if stats.get("last_pull") else None,
        }
