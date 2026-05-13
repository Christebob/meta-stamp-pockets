"""
META-STAMP V3 Wallet API Router Module.

This module provides FastAPI endpoints for wallet operations including
balance retrieval and transaction history management. Implements comprehensive
financial tracking for creator compensation through AI Touch Value™ calculations.

Endpoints:
    GET /balance - Retrieve current wallet balance with pending earnings
    GET /history - Retrieve paginated transaction history with filtering

Per Agent Action Plan sections 0.4, 0.6, 0.8, and 0.10 requirements for
wallet endpoints with authenticated user access and state management.
"""

import logging

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from app.core.auth import get_current_user
from app.core.database import get_db_client
from app.models.wallet import TransactionStatus, TransactionType


# Configure module logger for structured logging
logger = logging.getLogger(__name__)


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(tags=["wallet"])


# =============================================================================
# Response Models
# =============================================================================


class BalanceResponse(BaseModel):
    """
    Response model for wallet balance endpoint.

    Provides comprehensive financial summary including current balance,
    pending earnings from AI Touch Value™ calculations, and lifetime totals.

    Attributes:
        balance: Current available balance for withdrawal.
        currency: ISO 4217 currency code (e.g., USD).
        pending_earnings: AI Touch Value™ earnings not yet available.
        total_earned: Lifetime total earnings across all transactions.
        last_updated: Timestamp of last balance update (ISO 8601 format).
    """

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        },
    )

    balance: float = Field(
        ...,
        description="Current available balance for withdrawal",
        examples=[150.50],
    )
    currency: str = Field(
        ...,
        description="ISO 4217 currency code",
        examples=["USD"],
    )
    pending_earnings: float = Field(
        ...,
        description="AI Touch Value™ earnings not yet available",
        examples=[25.75],
    )
    total_earned: float = Field(
        ...,
        description="Lifetime total earnings",
        examples=[500.00],
    )
    last_updated: datetime = Field(
        ...,
        description="Timestamp of last balance update (ISO 8601)",
    )


class TransactionItem(BaseModel):
    """
    Simplified transaction representation for API responses.

    Provides essential transaction details for display in transaction history
    without exposing internal implementation details.

    Attributes:
        id: Unique transaction identifier.
        amount: Transaction amount (positive for earnings, negative for payouts).
        type: Transaction type (earning, payout, adjustment, refund).
        timestamp: Transaction creation timestamp (ISO 8601 format).
        description: Human-readable transaction description.
        status: Current transaction status.
    """

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        },
    )

    id: str = Field(
        ...,
        description="Unique transaction identifier",
        examples=["64a1b2c3d4e5f6g7h8i9j0k1"],
    )
    amount: float = Field(
        ...,
        description="Transaction amount",
        examples=[25.50],
    )
    type: str = Field(
        ...,
        description="Transaction type (earning, payout, adjustment, refund)",
        examples=["earning"],
    )
    timestamp: datetime = Field(
        ...,
        description="Transaction creation timestamp (ISO 8601)",
    )
    description: str = Field(
        ...,
        description="Human-readable transaction description",
        examples=["AI Touch Value™ earnings for asset"],
    )
    status: str = Field(
        ...,
        description="Current transaction status",
        examples=["completed"],
    )


class TransactionHistoryResponse(BaseModel):
    """
    Response model for transaction history endpoint.

    Provides paginated transaction list with metadata for navigation
    and filtering capabilities per Agent Action Plan section 0.8.

    Attributes:
        transactions: List of transaction items for current page.
        total: Total number of transactions matching filter criteria.
        skip: Number of transactions skipped (offset).
        limit: Maximum transactions per page.
    """

    transactions: list[TransactionItem] = Field(
        ...,
        description="List of transaction items for current page",
    )
    total: int = Field(
        ...,
        description="Total number of transactions matching filter criteria",
        ge=0,
        examples=[150],
    )
    skip: int = Field(
        ...,
        description="Number of transactions skipped (offset)",
        ge=0,
        examples=[0],
    )
    limit: int = Field(
        ...,
        description="Maximum transactions per page",
        ge=1,
        le=100,
        examples=[50],
    )


# =============================================================================
# Helper Functions
# =============================================================================


async def get_or_create_wallet(user_id: str) -> dict[str, Any]:
    """
    Retrieve or create a wallet for the specified user.

    If no wallet exists for the user, creates a new default wallet
    with zero balance. This ensures every authenticated user has
    a wallet record for tracking earnings.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        dict: The wallet document from MongoDB.

    Raises:
        HTTPException: With 500 status if database operation fails.
    """
    # Default wallet returned when database is unavailable
    now = datetime.now(UTC)
    default_wallet: dict[str, Any] = {
        "user_id": user_id,
        "balance": 0.0,
        "currency": "USD",
        "pending_earnings": 0.0,
        "total_earned": 0.0,
        "total_paid_out": 0.0,
        "last_payout_at": None,
        "created_at": now,
        "updated_at": now,
    }

    try:
        db_client = get_db_client()
        wallet_collection = db_client.get_wallet_collection()

        # Try to find existing wallet
        wallet = await wallet_collection.find_one({"user_id": user_id})

        if wallet is not None:
            logger.debug("Found existing wallet for user: %s", user_id)
            wallet_data: dict[str, Any] = dict(wallet) if wallet else {}
            return wallet_data

        # Create new default wallet for user
        logger.info("Creating new wallet for user: %s", user_id)
        new_wallet = dict(default_wallet)

        result = await wallet_collection.insert_one(new_wallet)
        new_wallet["_id"] = result.inserted_id

        logger.info(
            "Created new wallet for user %s with ID: %s",
            user_id,
            str(result.inserted_id),
        )
        return new_wallet

    except HTTPException:
        raise
    except RuntimeError:
        # Database not initialized — return a zero-balance default wallet
        # instead of crashing with 500/503. This keeps the UI functional
        # even when MongoDB is temporarily unavailable.
        logger.warning(
            "Database unavailable — returning default wallet for user: %s", user_id
        )
        return default_wallet
    except Exception as e:
        logger.exception("Unexpected error getting/creating wallet for user: %s", user_id)
        # Graceful degradation: return default wallet instead of 500
        logger.warning("Returning default wallet due to error for user: %s", user_id)
        return default_wallet


async def calculate_pending_earnings(user_id: str) -> float:
    """
    Calculate pending AI Touch Value™ earnings for a user.

    Queries the analytics collection to sum all AI Touch Value™
    calculations that have not yet been converted to wallet balance.
    Pending earnings represent compensation owed but not yet available
    for withdrawal.

    Args:
        user_id: The unique identifier of the user.

    Returns:
        Decimal: Total pending earnings amount.

    Raises:
        HTTPException: With 500 status if database operation fails.
    """
    try:
        db_client = get_db_client()
        analytics_collection = db_client.get_analytics_collection()

        # Aggregate pending AI Touch Value™ calculations
        # Status "pending" indicates not yet converted to wallet balance
        pipeline: list[dict[str, Any]] = [
            {
                "$match": {
                    "user_id": user_id,
                    "status": "pending",
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_pending": {"$sum": "$calculated_value"},
                }
            },
        ]

        cursor = analytics_collection.aggregate(pipeline)
        result = await cursor.to_list(length=1)

        if result and result[0].get("total_pending"):
            total = result[0]["total_pending"]
            logger.debug(
                "Calculated pending earnings for user %s: %s",
                user_id,
                total,
            )
            return float(total)

        return 0.0

    except RuntimeError:
        # Database not initialized — return zero pending instead of crashing
        logger.warning(
            "Database unavailable — returning 0 pending earnings for user: %s", user_id
        )
        return 0.0
    except Exception:
        logger.exception(
            "Unexpected error calculating pending earnings for user: %s",
            user_id,
        )
        # Graceful degradation: return 0 instead of 500
        return 0.0


async def get_transactions_collection() -> Any:
    """
    Get the transactions collection from MongoDB.

    The transactions collection stores individual financial events
    separately from wallet balance records, enabling detailed
    transaction history and auditing capabilities.

    Returns:
        AsyncIOMotorCollection: MongoDB collection for transactions.

    Raises:
        RuntimeError: If database is not connected.
    """
    db_client = get_db_client()
    database = db_client.get_database()
    return database["transactions"]


# =============================================================================
# API Endpoints
# =============================================================================


@router.get(
    "/balance",
    response_model=BalanceResponse,
    summary="Get wallet balance",
    description="Retrieve current wallet balance including pending AI Touch Value™ earnings "
    "and lifetime totals for the authenticated user.",
    responses={
        200: {
            "description": "Successfully retrieved wallet balance",
            "model": BalanceResponse,
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        500: {
            "description": "Internal server error - Database operation failed",
        },
        503: {
            "description": "Service unavailable - Database connection failed",
        },
    },
)
async def get_balance(
    current_user: dict[str, Any] = Depends(get_current_user),
) -> BalanceResponse:
    """
    Retrieve current wallet balance for the authenticated user.

    Returns comprehensive financial summary including:
    - Current available balance for withdrawal
    - Pending AI Touch Value™ earnings not yet available
    - Lifetime total earnings across all transactions
    - Last balance update timestamp

    The balance calculation combines stored wallet balance with
    dynamically calculated pending earnings from the analytics
    collection to provide real-time financial status.

    Args:
        current_user: Authenticated user from JWT token (injected via Depends).

    Returns:
        BalanceResponse: Complete wallet balance summary.

    Raises:
        HTTPException: With 401 if unauthorized, 500/503 for server errors.
    """
    # Extract user ID from authenticated user document
    user_id = current_user.get("_id")
    if not user_id:
        logger.error("User document missing _id field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication",
        )

    # Convert ObjectId to string if needed
    if isinstance(user_id, ObjectId):
        user_id = str(user_id)

    logger.info("Retrieving wallet balance for user: %s", user_id)

    try:
        # Get or create wallet for user
        wallet = await get_or_create_wallet(user_id)

        # Calculate pending earnings from analytics
        pending_from_analytics = await calculate_pending_earnings(user_id)

        # Combine wallet pending_earnings with newly calculated pending
        wallet_pending = float(wallet.get("pending_earnings", 0.0) or 0.0)
        pending_from_analytics_f = float(pending_from_analytics)
        total_pending = max(wallet_pending, pending_from_analytics_f)

        # Extract balance values
        balance = float(wallet.get("balance", 0.0) or 0.0)
        total_earned = float(wallet.get("total_earned", 0.0) or 0.0)

        currency = wallet.get("currency", "USD")
        updated_at = wallet.get("updated_at", datetime.now(UTC))

        logger.info(
            "Wallet balance for user %s: balance=%s, pending=%s, total_earned=%s",
            user_id,
            balance,
            total_pending,
            total_earned,
        )

        return BalanceResponse(
            balance=balance,
            currency=currency,
            pending_earnings=total_pending,
            total_earned=total_earned,
            last_updated=updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error retrieving balance for user: %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve wallet balance",
        ) from e


@router.get(
    "/history",
    response_model=TransactionHistoryResponse,
    summary="Get transaction history",
    description="Retrieve paginated transaction history with optional filtering "
    "by transaction type for the authenticated user.",
    responses={
        200: {
            "description": "Successfully retrieved transaction history",
            "model": TransactionHistoryResponse,
        },
        400: {
            "description": "Bad request - Invalid query parameters",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        500: {
            "description": "Internal server error - Database operation failed",
        },
        503: {
            "description": "Service unavailable - Database connection failed",
        },
    },
)
async def get_history(
    current_user: dict[str, Any] = Depends(get_current_user),
    skip: int = Query(
        default=0,
        ge=0,
        description="Number of transactions to skip (offset for pagination)",
        examples=[0],
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
        description="Maximum number of transactions to return per page (1-100)",
        examples=[50],
    ),
    transaction_type: str | None = Query(
        default=None,
        description="Filter by transaction type: earning, payout, adjustment, refund",
        examples=["earning"],
    ),
) -> TransactionHistoryResponse:
    """
    Retrieve paginated transaction history for the authenticated user.

    Returns a paginated list of transactions sorted by timestamp in
    descending order (newest first). Supports filtering by transaction
    type to show specific categories of financial events.

    Transaction types:
    - earning: Income from AI Touch Value™ calculations
    - payout: Withdrawals to external payment methods
    - adjustment: Manual balance corrections
    - refund: Return of previously paid funds

    Args:
        current_user: Authenticated user from JWT token (injected via Depends).
        skip: Number of transactions to skip for pagination (default: 0).
        limit: Maximum transactions per page (default: 50, max: 100).
        transaction_type: Optional filter for transaction type.

    Returns:
        TransactionHistoryResponse: Paginated transaction list with metadata.

    Raises:
        HTTPException: With 400 for invalid params, 401 if unauthorized,
                      500/503 for server errors.
    """
    # Extract user ID from authenticated user document
    user_id = current_user.get("_id")
    if not user_id:
        logger.error("User document missing _id field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication",
        )

    # Convert ObjectId to string if needed
    if isinstance(user_id, ObjectId):
        user_id = str(user_id)

    # Validate transaction_type if provided
    valid_types = [t.value for t in TransactionType]
    if transaction_type is not None:
        transaction_type_lower = transaction_type.lower()
        if transaction_type_lower not in valid_types:
            logger.warning(
                "Invalid transaction type requested: %s (valid: %s)",
                transaction_type,
                valid_types,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid transaction_type. Must be one of: {', '.join(valid_types)}",
            )
        transaction_type = transaction_type_lower

    logger.info(
        "Retrieving transaction history for user: %s (skip=%d, limit=%d, type=%s)",
        user_id,
        skip,
        limit,
        transaction_type,
    )

    try:
        # Build query filter
        query_filter: dict[str, Any] = {"user_id": user_id}

        if transaction_type:
            query_filter["transaction_type"] = transaction_type

        # Get transactions collection
        transactions_collection = await get_transactions_collection()

        # Get total count for pagination
        total_count = await transactions_collection.count_documents(query_filter)

        # Query transactions with sorting and pagination
        cursor = (
            transactions_collection.find(query_filter)
            .sort("created_at", -1)  # Newest first
            .skip(skip)
            .limit(limit)
        )

        transactions_docs = await cursor.to_list(length=limit)

        # Transform documents to TransactionItem format
        transactions: list[TransactionItem] = []
        for doc in transactions_docs:
            # Extract and convert fields
            doc_id = doc.get("_id")
            if isinstance(doc_id, ObjectId):
                doc_id = str(doc_id)

            amount = doc.get("amount", Decimal("0.00"))
            if isinstance(amount, Decimal):
                amount = float(amount)
            elif not isinstance(amount, (int, float)):
                amount = float(Decimal(str(amount)))

            tx_type = doc.get("transaction_type", "earning")
            if isinstance(tx_type, TransactionType):
                tx_type = tx_type.value

            tx_status = doc.get("status", "completed")
            if isinstance(tx_status, TransactionStatus):
                tx_status = tx_status.value

            timestamp = doc.get("created_at", datetime.now(UTC))
            description = doc.get("description", "Transaction")

            transactions.append(
                TransactionItem(
                    id=doc_id or "",
                    amount=amount,
                    type=tx_type,
                    timestamp=timestamp,
                    description=description,
                    status=tx_status,
                )
            )

        logger.info(
            "Retrieved %d transactions for user %s (total: %d)",
            len(transactions),
            user_id,
            total_count,
        )

        return TransactionHistoryResponse(
            transactions=transactions,
            total=total_count,
            skip=skip,
            limit=limit,
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.exception("Database unavailable while retrieving history")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable",
        ) from e
    except Exception as e:
        logger.exception(
            "Unexpected error retrieving transaction history for user: %s",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction history",
        ) from e


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "BalanceResponse",
    "TransactionHistoryResponse",
    "TransactionItem",
    "calculate_pending_earnings",
    "get_or_create_wallet",
    "router",
]
