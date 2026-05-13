"""
Wallet Financial Tracking Models for META-STAMP V3.

This module defines Pydantic models for comprehensive financial tracking including
WalletBalance for user balance management and Transaction for recording all
financial events. Supports AI Touch Value™ earnings tracking, payout management,
and maintains a complete transaction history for creator compensation.

All monetary values use Decimal type to ensure precision and avoid floating-point
errors in financial calculations per Agent Action Plan section 0.3 requirements.
"""

from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Constants
# =============================================================================

# Minimum payout threshold in USD
MINIMUM_PAYOUT_THRESHOLD = Decimal("10.00")

# Valid ISO 4217 currency codes supported by the platform
SUPPORTED_CURRENCIES = {"USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF"}

# Decimal precision for monetary values (2 decimal places)
DECIMAL_PRECISION = Decimal("0.01")


# =============================================================================
# Enumerations
# =============================================================================


class TransactionType(str, Enum):
    """
    Enumeration of valid transaction types for wallet operations.

    Provides type-safe transaction categorization for proper accounting
    and financial reporting in the creator compensation system.

    Attributes:
        EARNING: Income from AI Touch Value™ calculations or content usage.
        PAYOUT: Withdrawal of funds to external payment method.
        ADJUSTMENT: Manual balance correction by administrators.
        REFUND: Return of previously paid out funds.
        PULL: Per-pull micro-payment from AI agent to creator.
    """

    EARNING = "earning"
    PAYOUT = "payout"
    ADJUSTMENT = "adjustment"
    REFUND = "refund"
    PULL = "pull"


class TransactionStatus(str, Enum):
    """
    Enumeration of valid transaction status values.

    Tracks the lifecycle state of each transaction from initiation
    through completion or failure, enabling accurate financial reporting.

    Attributes:
        PENDING: Transaction initiated but not yet processed.
        COMPLETED: Transaction successfully processed and finalized.
        FAILED: Transaction failed during processing.
        CANCELLED: Transaction cancelled before completion.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# Transaction Model
# =============================================================================


class Transaction(BaseModel):
    """
    Pydantic model for individual financial transaction records.

    Represents a single financial event in the system including earnings
    from AI Touch Value™ calculations, payouts to creators, administrative
    adjustments, and refunds. Maintains comprehensive audit trail with
    timestamps and metadata for financial compliance.

    Attributes:
        id: MongoDB ObjectId as string (aliased from _id).
        user_id: ID of the user associated with this transaction.
        transaction_type: Type of transaction (earning, payout, adjustment, refund).
        amount: Transaction amount with 2 decimal precision.
        currency: ISO 4217 currency code (default: USD).
        status: Current status of the transaction.
        description: Human-readable description of the transaction.
        asset_id: Optional ID of associated asset for earnings attribution.
        reference_id: Optional external reference for tracking.
        created_at: UTC timestamp when transaction was created.
        completed_at: UTC timestamp when transaction was completed.
        metadata: Additional transaction details as key-value pairs.

    Example:
        >>> transaction = Transaction(
        ...     user_id="user123",
        ...     transaction_type=TransactionType.EARNING,
        ...     amount=Decimal("25.50"),
        ...     status=TransactionStatus.COMPLETED,
        ...     description="AI Touch Value™ earnings for asset",
        ...     asset_id="asset456"
        ... )
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            Decimal: lambda v: str(v.quantize(DECIMAL_PRECISION)),
            datetime: lambda v: v.isoformat(),
        },
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Core identification fields
    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")
    user_id: str = Field(
        ..., min_length=1, max_length=100, description="User ID associated with this transaction"
    )

    # Transaction details
    transaction_type: TransactionType = Field(
        ..., description="Type of transaction (earning, payout, adjustment, refund)"
    )
    amount: Decimal = Field(..., description="Transaction amount with 2 decimal precision")
    currency: str = Field(
        default="USD", min_length=3, max_length=3, description="ISO 4217 currency code"
    )
    status: TransactionStatus = Field(
        default=TransactionStatus.PENDING, description="Current transaction status"
    )
    description: str = Field(
        ..., min_length=1, max_length=500, description="Human-readable transaction description"
    )

    # Optional associations
    asset_id: str | None = Field(
        default=None, max_length=100, description="Associated asset ID for earnings attribution"
    )
    reference_id: str | None = Field(
        default=None, max_length=255, description="External reference ID for tracking"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when transaction was created",
    )
    completed_at: datetime | None = Field(
        default=None, description="UTC timestamp when transaction was completed"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional transaction details"
    )

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, value: Decimal) -> Decimal:
        """
        Validate transaction amount ensuring proper decimal format.

        For EARNING transactions, amount must be positive.
        For PAYOUT transactions, amount must be positive (represents outflow).
        For ADJUSTMENT and REFUND, amount can be positive or negative.

        Args:
            value: The amount to validate.

        Returns:
            Validated and quantized Decimal amount.

        Raises:
            ValueError: If amount format is invalid.
        """
        try:
            # Ensure value is a Decimal
            if not isinstance(value, Decimal):
                value = Decimal(str(value))

            # Quantize to 2 decimal places for monetary precision
            return value.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)

        except (InvalidOperation, TypeError) as e:
            raise ValueError(f"Invalid amount format: {e}") from e

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        """
        Validate currency code against ISO 4217 standards.

        Ensures the currency code is uppercase and within the
        supported currencies for the platform.

        Args:
            value: The currency code to validate.

        Returns:
            Uppercase currency code.

        Raises:
            ValueError: If currency code is not supported.
        """
        value = value.upper().strip()

        if value not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Unsupported currency code: {value}. "
                f"Supported currencies: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
            )

        return value


# =============================================================================
# WalletBalance Model
# =============================================================================


class WalletBalance(BaseModel):
    """
    Pydantic model for user wallet balance tracking.

    Maintains comprehensive financial state for a user including current
    balance, pending earnings from AI Touch Value™ calculations, lifetime
    totals, and payout history. Provides helper methods for determining
    available balance and payout eligibility.

    Attributes:
        id: MongoDB ObjectId as string (aliased from _id).
        user_id: Unique user ID (unique constraint in MongoDB).
        balance: Current available balance with 2 decimal precision.
        currency: ISO 4217 currency code (default: USD).
        pending_earnings: Pending AI Touch Value™ amounts not yet available.
        total_earned: Lifetime total earnings.
        total_paid_out: Lifetime total payouts.
        last_payout_at: Timestamp of most recent payout.
        created_at: UTC timestamp when wallet was created.
        updated_at: UTC timestamp of last wallet update.

    Example:
        >>> wallet = WalletBalance(
        ...     user_id="user123",
        ...     balance=Decimal("150.00"),
        ...     pending_earnings=Decimal("25.50"),
        ...     total_earned=Decimal("500.00"),
        ...     total_paid_out=Decimal("350.00")
        ... )
        >>> wallet.calculate_available_balance()
        Decimal('150.00')
        >>> wallet.is_payout_eligible()
        True
    """

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={
            Decimal: lambda v: str(v.quantize(DECIMAL_PRECISION)),
            datetime: lambda v: v.isoformat(),
        },
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Core identification fields
    id: str | None = Field(default=None, alias="_id", description="MongoDB ObjectId as string")
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique user ID (unique constraint in MongoDB)",
    )

    # Balance fields with precise decimal handling
    balance: Decimal = Field(default=Decimal("0.00"), description="Current available balance")
    currency: str = Field(
        default="USD", min_length=3, max_length=3, description="ISO 4217 currency code"
    )
    pending_earnings: Decimal = Field(
        default=Decimal("0.00"), description="Pending AI Touch Value™ earnings"
    )
    total_earned: Decimal = Field(default=Decimal("0.00"), description="Lifetime total earnings")
    total_paid_out: Decimal = Field(default=Decimal("0.00"), description="Lifetime total payouts")

    # Payout tracking
    last_payout_at: datetime | None = Field(
        default=None, description="Timestamp of most recent payout"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when wallet was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="UTC timestamp of last wallet update"
    )

    @field_validator("balance", "pending_earnings", "total_earned", "total_paid_out")
    @classmethod
    def validate_monetary_fields(cls, value: Decimal) -> Decimal:
        """
        Validate monetary fields ensuring proper decimal format.

        All monetary fields must be non-negative and properly quantized
        to 2 decimal places for accurate financial tracking.

        Args:
            value: The monetary value to validate.

        Returns:
            Validated and quantized Decimal value.

        Raises:
            ValueError: If value is negative or invalid format.
        """
        try:
            # Ensure value is a Decimal
            if not isinstance(value, Decimal):
                value = Decimal(str(value))

            # Quantize to 2 decimal places
            quantized_value = value.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)

            # Monetary balances must be non-negative
            if quantized_value < Decimal("0.00"):
                raise ValueError("Monetary values cannot be negative")

            return quantized_value
        except (InvalidOperation, TypeError) as e:
            raise ValueError(f"Invalid monetary value format: {e}") from e

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str) -> str:
        """
        Validate currency code against ISO 4217 standards.

        Ensures the currency code is uppercase and within the
        supported currencies for the platform.

        Args:
            value: The currency code to validate.

        Returns:
            Uppercase currency code.

        Raises:
            ValueError: If currency code is not supported.
        """
        value = value.upper().strip()

        if value not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"Unsupported currency code: {value}. "
                f"Supported currencies: {', '.join(sorted(SUPPORTED_CURRENCIES))}"
            )

        return value

    def calculate_available_balance(self) -> Decimal:
        """
        Calculate the currently available balance for withdrawal.

        The available balance is the confirmed balance that can be
        withdrawn by the user. Pending earnings are not included
        as they require confirmation before becoming available.

        Returns:
            Decimal: The available balance for withdrawal.

        Example:
            >>> wallet = WalletBalance(user_id="user123", balance=Decimal("100.00"))
            >>> wallet.calculate_available_balance()
            Decimal('100.00')
        """
        return self.balance.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)

    def is_payout_eligible(self, minimum_threshold: Decimal | None = None) -> bool:
        """
        Check if the wallet meets the minimum payout threshold.

        Determines whether the user can request a payout based on
        their available balance meeting or exceeding the minimum
        payout threshold (default: $10.00 USD).

        Args:
            minimum_threshold: Custom minimum threshold. If None,
                uses the default MINIMUM_PAYOUT_THRESHOLD.

        Returns:
            bool: True if balance meets threshold, False otherwise.

        Example:
            >>> wallet = WalletBalance(user_id="user123", balance=Decimal("15.00"))
            >>> wallet.is_payout_eligible()
            True
            >>> wallet.balance = Decimal("5.00")
            >>> wallet.is_payout_eligible()
            False
            >>> wallet.is_payout_eligible(minimum_threshold=Decimal("3.00"))
            True
        """
        threshold = minimum_threshold if minimum_threshold is not None else MINIMUM_PAYOUT_THRESHOLD
        available = self.calculate_available_balance()
        return available >= threshold


# =============================================================================
# Export all models for convenient importing
# =============================================================================

__all__ = [
    "MINIMUM_PAYOUT_THRESHOLD",
    "SUPPORTED_CURRENCIES",
    "Transaction",
    "TransactionStatus",
    "TransactionType",
    "WalletBalance",
]
