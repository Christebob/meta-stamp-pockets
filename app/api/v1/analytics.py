"""
META-STAMP V3 Analytics API Router Module.

This module implements the AI Touch Value™ calculation endpoint for determining
creator compensation based on their content's contribution to AI model training.

The calculation follows the exact formula specified in the Agent Action Plan:

    AI Touch Value(TM) = ModelEarnings x (TrainingContributionScore/100)
                         x (UsageExposureScore/100) x 0.25

Where:
    - ModelEarnings: Total earnings from the AI model in USD (≥ 0)
    - TrainingContributionScore: Score 0-100 representing data contribution
    - UsageExposureScore: Score 0-100 representing content exposure/usage
    - EquityFactor: Fixed at 0.25 (25%) - NON-NEGOTIABLE per requirements

Endpoints:
    - POST /predict: Calculate AI Touch Value™ from provided metrics

Per Agent Action Plan sections 0.3, 0.4, 0.6, 0.8, and 0.10:
    - EquityFactor MUST be fixed at 0.25 (25%)
    - Input validation: model_earnings ≥ 0, scores 0-100
    - Store calculation history in MongoDB analytics collection
    - Return detailed response with formula breakdown

Example:
    >>> # Request
    >>> POST /api/v1/analytics/predict
    >>> {
    >>>     "model_earnings": 10000.0,
    >>>     "training_contribution_score": 75.0,
    >>>     "usage_exposure_score": 80.0
    >>> }
    >>>
    >>> # Response
    >>> {
    >>>     "calculated_value": 1500.0,
    >>>     "equity_factor": 0.25,
    >>>     "formula_breakdown": "$10,000.00 x (75.0/100) x (80.0/100) x 0.25 = $1,500.00"
    >>> }
"""

import logging

from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.core.auth import get_current_user
from app.core.database import get_database
from app.models.analytics import AITouchValueCalculation
from app.services.ai_value_service import AIValueService


# Configure module logger for structured logging
logger = logging.getLogger(__name__)


# =============================================================================
# Router Configuration
# =============================================================================

router = APIRouter(
    tags=["analytics"],
    responses={
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid input parameters",
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Calculation or database error",
        },
    },
)


# =============================================================================
# Constants
# =============================================================================

# Fixed equity factor per Agent Action Plan section 0.3
EQUITY_FACTOR: float = 0.25

# Valid score range (0-100)
SCORE_MIN: float = 0.0
SCORE_MAX: float = 100.0

# Pagination limits
PAGINATION_MAX_LIMIT: int = 100


# =============================================================================
# Request/Response Models
# =============================================================================


class PredictRequest(BaseModel):
    """
    Request model for AI Touch Value™ prediction endpoint.

    This model validates all input parameters according to the AI Touch Value™
    formula requirements:
    - model_earnings: Must be non-negative (≥ 0)
    - training_contribution_score: Must be in range 0-100
    - usage_exposure_score: Must be in range 0-100
    - asset_id: Optional asset identifier for asset-specific calculations

    Attributes:
        model_earnings: AI model earnings in USD (must be ≥ 0)
        training_contribution_score: Training data contribution score (0-100)
        usage_exposure_score: Usage/exposure score (0-100)
        asset_id: Optional associated asset ID for linking calculation to asset

    Example:
        >>> request = PredictRequest(
        ...     model_earnings=10000.0,
        ...     training_contribution_score=75.0,
        ...     usage_exposure_score=80.0
        ... )
    """

    model_earnings: float = Field(
        ...,
        ge=0,
        description="Model earnings in USD. Must be non-negative.",
        json_schema_extra={"example": 10000.0},
    )
    training_contribution_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Training contribution score from 0-100 representing data contribution to AI training.",
        json_schema_extra={"example": 75.0},
    )
    usage_exposure_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Usage exposure score from 0-100 representing content exposure/usage.",
        json_schema_extra={"example": 80.0},
    )
    asset_id: str | None = Field(
        default=None,
        description="Optional associated asset ID for linking calculation to a specific asset.",
        json_schema_extra={"example": "507f1f77bcf86cd799439011"},
    )

    @field_validator("model_earnings", mode="before")
    @classmethod
    def validate_model_earnings(cls, value: Any) -> float:
        """
        Validate that model_earnings is a valid non-negative number.

        Args:
            value: Input value to validate

        Returns:
            float: Validated model earnings value

        Raises:
            ValueError: If value is negative or not a valid number
        """
        if value is None:
            raise ValueError("model_earnings is required and cannot be None")

        try:
            earnings = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"model_earnings must be a valid number, got {type(value).__name__}"
            ) from e

        if earnings < 0:
            raise ValueError(f"model_earnings must be non-negative (≥ 0), got {earnings}")

        return earnings

    @field_validator("training_contribution_score", mode="before")
    @classmethod
    def validate_contribution_score(cls, value: Any) -> float:
        """
        Validate that training_contribution_score is within 0-100 range.

        Args:
            value: Input score value

        Returns:
            float: Validated contribution score

        Raises:
            ValueError: If score is outside 0-100 range or not a valid number
        """
        if value is None:
            raise ValueError("training_contribution_score is required")

        try:
            score = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"training_contribution_score must be a valid number, got {type(value).__name__}"
            ) from e

        if score < SCORE_MIN or score > SCORE_MAX:
            raise ValueError(
                f"training_contribution_score must be between {SCORE_MIN} and {SCORE_MAX}, got {score}"
            )

        return score

    @field_validator("usage_exposure_score", mode="before")
    @classmethod
    def validate_exposure_score(cls, value: Any) -> float:
        """
        Validate that usage_exposure_score is within 0-100 range.

        Args:
            value: Input score value

        Returns:
            float: Validated exposure score

        Raises:
            ValueError: If score is outside 0-100 range or not a valid number
        """
        if value is None:
            raise ValueError("usage_exposure_score is required")

        try:
            score = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"usage_exposure_score must be a valid number, got {type(value).__name__}"
            ) from e

        if score < SCORE_MIN or score > SCORE_MAX:
            raise ValueError(
                f"usage_exposure_score must be between {SCORE_MIN} and {SCORE_MAX}, got {score}"
            )

        return score


class PredictResponse(BaseModel):
    """
    Response model for AI Touch Value™ prediction endpoint.

    Contains the calculated AI Touch Value™ along with all input parameters,
    formula breakdown for transparency, and metadata for audit purposes.

    Attributes:
        calculated_value: The computed AI Touch Value™ result in USD
        equity_factor: Fixed equity factor (always 0.25)
        model_earnings: Input model earnings used in calculation
        training_contribution_score: Input training contribution score
        usage_exposure_score: Input usage exposure score
        calculation_id: MongoDB ObjectId of stored calculation record
        timestamp: ISO format UTC timestamp of calculation
        formula_breakdown: Human-readable formula breakdown string
        asset_id: Associated asset ID if provided in request

    Example:
        >>> response = PredictResponse(
        ...     calculated_value=1500.0,
        ...     equity_factor=0.25,
        ...     model_earnings=10000.0,
        ...     training_contribution_score=75.0,
        ...     usage_exposure_score=80.0,
        ...     calculation_id="507f1f77bcf86cd799439011",
        ...     timestamp="2025-01-15T10:30:00Z",
        ...     formula_breakdown="$10,000.00 x (75.0/100) x (80.0/100) x 0.25 = $1,500.00"
        ... )
    """

    calculated_value: float = Field(
        ...,
        description="The computed AI Touch Value™ result in USD",
        json_schema_extra={"example": 1500.0},
    )
    equity_factor: float = Field(
        default=EQUITY_FACTOR,
        description="Fixed equity factor at 0.25 (25%)",
        json_schema_extra={"example": 0.25},
    )
    model_earnings: float = Field(
        ...,
        description="Input model earnings used in calculation",
        json_schema_extra={"example": 10000.0},
    )
    training_contribution_score: float = Field(
        ...,
        description="Input training contribution score (0-100)",
        json_schema_extra={"example": 75.0},
    )
    usage_exposure_score: float = Field(
        ...,
        description="Input usage exposure score (0-100)",
        json_schema_extra={"example": 80.0},
    )
    calculation_id: str | None = Field(
        default=None,
        description="MongoDB ObjectId of stored calculation record",
        json_schema_extra={"example": "507f1f77bcf86cd799439011"},
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO format UTC timestamp of calculation",
        json_schema_extra={"example": "2025-01-15T10:30:00+00:00"},
    )
    formula_breakdown: str = Field(
        ...,
        description="Human-readable formula breakdown for transparency and auditing",
        json_schema_extra={"example": "$10,000.00 x (75.0/100) x (80.0/100) x 0.25 = $1,500.00"},
    )
    asset_id: str | None = Field(
        default=None,
        description="Associated asset ID if provided in request",
        json_schema_extra={"example": "507f1f77bcf86cd799439011"},
    )
    user_id: str | None = Field(
        default=None,
        description="User ID who performed the calculation",
        json_schema_extra={"example": "507f1f77bcf86cd799439011"},
    )


class ErrorResponse(BaseModel):
    """
    Error response model for standardized error handling.

    Attributes:
        detail: Human-readable error description
        error_code: Machine-readable error code for client handling
        field: Optional field name that caused the error
    """

    detail: str = Field(
        ...,
        description="Human-readable error description",
    )
    error_code: str = Field(
        default="CALCULATION_ERROR",
        description="Machine-readable error code",
    )
    field: str | None = Field(
        default=None,
        description="Field name that caused the error (if applicable)",
    )


# =============================================================================
# Utility Functions
# =============================================================================


def format_currency(value: float) -> str:
    """
    Format a numeric value as USD currency string.

    Args:
        value: Numeric value to format

    Returns:
        str: Formatted currency string (e.g., "$1,500.00")
    """
    return f"${value:,.2f}"


def generate_formula_breakdown(
    model_earnings: float,
    contribution_score: float,
    exposure_score: float,
    calculated_value: float,
) -> str:
    """
    Generate a human-readable formula breakdown string.

    Creates a transparent breakdown of the AI Touch Value™ calculation
    showing all inputs, factors, and the final result for audit purposes.

    Args:
        model_earnings: Model earnings input value
        contribution_score: Training contribution score (0-100)
        exposure_score: Usage exposure score (0-100)
        calculated_value: The computed AI Touch Value™

    Returns:
        str: Human-readable formula breakdown string

    Example:
        >>> breakdown = generate_formula_breakdown(10000.0, 75.0, 80.0, 1500.0)
        >>> print(breakdown)
        "$10,000.00 x (75.0/100) x (80.0/100) x 0.25 = $1,500.00"
    """
    return (
        f"{format_currency(model_earnings)} x "
        f"({contribution_score}/100) x "
        f"({exposure_score}/100) x "
        f"{EQUITY_FACTOR} = "
        f"{format_currency(calculated_value)}"
    )


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate AI Touch Value™",
    description=(
        "Calculate the AI Touch Value(TM) compensation estimate based on the formula: "
        "AI Touch Value(TM) = ModelEarnings x (TrainingContributionScore/100) x "
        "(UsageExposureScore/100) x 0.25. The equity factor is fixed at 25%."
    ),
    responses={
        status.HTTP_200_OK: {
            "description": "Calculation successful",
            "model": PredictResponse,
        },
        status.HTTP_400_BAD_REQUEST: {
            "description": "Invalid input parameters",
            "model": ErrorResponse,
        },
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Authentication required",
        },
        422: {
            "description": "Validation error",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Calculation or database error",
            "model": ErrorResponse,
        },
    },
)
async def predict_ai_touch_value(
    request: PredictRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """
    Calculate AI Touch Value(TM) using the exact formula.

    This endpoint calculates the AI Touch Value(TM) compensation estimate for creators
    based on their content's contribution to AI model training. The calculation
    follows the exact formula specified in the Agent Action Plan:

        AI Touch Value(TM) = ModelEarnings x (TrainingContributionScore/100)
                             x (UsageExposureScore/100) x 0.25

    The calculation is stored in MongoDB for historical tracking and trend analysis.

    Args:
        request: PredictRequest containing model_earnings, scores, and optional asset_id
        current_user: Authenticated user from get_current_user dependency

    Returns:
        JSONResponse: PredictResponse with calculated value, breakdown, and metadata

    Raises:
        HTTPException: 400 for invalid inputs, 401 for unauthorized, 500 for errors

    Security:
        - Requires valid JWT authentication
        - User ID is extracted from the authenticated token
        - Calculation is associated with the authenticated user

    Example Request:
        POST /api/v1/analytics/predict
        Authorization: Bearer <token>
        Content-Type: application/json
        {
            "model_earnings": 10000.0,
            "training_contribution_score": 75.0,
            "usage_exposure_score": 80.0,
            "asset_id": "507f1f77bcf86cd799439011"
        }

    Example Response:
        {
            "calculated_value": 1500.0,
            "equity_factor": 0.25,
            "model_earnings": 10000.0,
            "training_contribution_score": 75.0,
            "usage_exposure_score": 80.0,
            "calculation_id": "507f1f77bcf86cd799439012",
            "timestamp": "2025-01-15T10:30:00+00:00",
            "formula_breakdown": "$10,000.00 x (75.0/100) x (80.0/100) x 0.25 = $1,500.00",
            "asset_id": "507f1f77bcf86cd799439011",
            "user_id": "507f1f77bcf86cd799439010"
        }
    """
    # Extract user ID from authenticated user
    user_id = current_user.get("_id")
    if not user_id:
        logger.error("Authenticated user missing _id field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    logger.info(
        f"AI Touch Value™ prediction requested by user={user_id}, "
        f"earnings=${request.model_earnings}, "
        f"contribution={request.training_contribution_score}, "
        f"exposure={request.usage_exposure_score}, "
        f"asset_id={request.asset_id}"
    )

    try:
        # Initialize the AI Value Service
        ai_value_service = AIValueService()

        # Call the service to calculate AI Touch Value™
        # The service handles validation, calculation, and storage
        calculation_result = await ai_value_service.calculate_ai_touch_value(
            model_earnings=request.model_earnings,
            contribution_score=request.training_contribution_score,
            exposure_score=request.usage_exposure_score,
            user_id=user_id,
            asset_id=request.asset_id,
        )

        # Parse the calculated value (service returns as string for precision)
        calculated_value = float(calculation_result["calculated_value"])

        # Generate human-readable formula breakdown
        formula_breakdown = generate_formula_breakdown(
            model_earnings=request.model_earnings,
            contribution_score=request.training_contribution_score,
            exposure_score=request.usage_exposure_score,
            calculated_value=calculated_value,
        )

        # Build response
        response_data = PredictResponse(
            calculated_value=calculated_value,
            equity_factor=EQUITY_FACTOR,
            model_earnings=request.model_earnings,
            training_contribution_score=request.training_contribution_score,
            usage_exposure_score=request.usage_exposure_score,
            calculation_id=calculation_result.get("calculation_id"),
            timestamp=calculation_result.get("timestamp", datetime.now(UTC).isoformat()),
            formula_breakdown=formula_breakdown,
            asset_id=request.asset_id,
            user_id=user_id,
        )

        logger.info(
            f"AI Touch Value™ calculated successfully: "
            f"${calculated_value:,.2f} for user={user_id}"
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data.model_dump(mode="json"),
        )

    except ValueError as e:
        # Input validation errors from the service
        logger.warning(f"AI Touch Value™ validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except RuntimeError:
        # Database not initialized or unavailable
        logger.exception("Database error during AI Touch Value(TM) calculation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please try again later.",
        ) from None

    except Exception:
        # Unexpected errors
        logger.exception("Unexpected error during AI Touch Value(TM) calculation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during calculation. Please try again.",
        ) from None


@router.get(
    "/history",
    response_model=list[PredictResponse],
    status_code=status.HTTP_200_OK,
    summary="Get calculation history",
    description="Retrieve the user's AI Touch Value™ calculation history for trend analysis.",
)
async def get_calculation_history(
    limit: int = 10,
    offset: int = 0,
    asset_id: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """
    Retrieve the user's AI Touch Value™ calculation history.

    Returns a paginated list of previous AI Touch Value™ calculations
    for the authenticated user, enabling trend analysis and historical review.

    Args:
        limit: Maximum number of records to return (default: 10, max: 100)
        offset: Number of records to skip for pagination (default: 0)
        asset_id: Optional filter by asset ID
        current_user: Authenticated user from get_current_user dependency

    Returns:
        JSONResponse: List of PredictResponse objects representing historical calculations

    Raises:
        HTTPException: 401 for unauthorized, 500 for database errors

    Example Request:
        GET /api/v1/analytics/history?limit=5&offset=0
        Authorization: Bearer <token>

    Example Response:
        [
            {
                "calculated_value": 1500.0,
                "equity_factor": 0.25,
                "model_earnings": 10000.0,
                ...
            },
            ...
        ]
    """
    # Extract user ID from authenticated user
    user_id = current_user.get("_id")
    if not user_id:
        logger.error("Authenticated user missing _id field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    # Validate pagination parameters
    if limit < 1 or limit > PAGINATION_MAX_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Limit must be between 1 and {PAGINATION_MAX_LIMIT}",
        )
    if offset < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offset must be non-negative",
        )

    logger.info(
        f"Fetching calculation history for user={user_id}, "
        f"limit={limit}, offset={offset}, asset_id={asset_id}"
    )

    try:
        # Get database and analytics collection
        database = get_database()
        analytics_collection = database["analytics"]

        # Build query filter
        query_filter: dict[str, Any] = {"user_id": user_id}
        if asset_id:
            query_filter["asset_id"] = asset_id

        # Query with pagination, sorted by created_at descending
        cursor = (
            analytics_collection.find(query_filter).sort("created_at", -1).skip(offset).limit(limit)
        )

        # Convert cursor to list
        records = await cursor.to_list(length=limit)

        # Transform records to response format
        history: list[dict[str, Any]] = []
        for record in records:
            # Parse model from MongoDB dict
            calc = AITouchValueCalculation.from_mongodb_dict(record)

            # Generate formula breakdown
            calculated_value = float(calc.calculated_value) if calc.calculated_value else 0.0
            formula_breakdown = generate_formula_breakdown(
                model_earnings=float(calc.model_earnings),
                contribution_score=calc.training_contribution_score,
                exposure_score=calc.usage_exposure_score,
                calculated_value=calculated_value,
            )

            history.append(
                {
                    "calculated_value": calculated_value,
                    "equity_factor": calc.equity_factor,
                    "model_earnings": float(calc.model_earnings),
                    "training_contribution_score": calc.training_contribution_score,
                    "usage_exposure_score": calc.usage_exposure_score,
                    "calculation_id": calc.id,
                    "timestamp": calc.created_at.isoformat() if calc.created_at else None,
                    "formula_breakdown": formula_breakdown,
                    "asset_id": calc.asset_id,
                    "user_id": calc.user_id,
                }
            )

        logger.info(f"Retrieved {len(history)} calculation records for user={user_id}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=history,
        )

    except RuntimeError:
        logger.exception("Database error fetching calculation history")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please try again later.",
        ) from None

    except Exception:
        logger.exception("Unexpected error fetching calculation history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from None


@router.get(
    "/history/{calculation_id}",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Get specific calculation",
    description="Retrieve a specific AI Touch Value™ calculation by ID.",
)
async def get_calculation_by_id(
    calculation_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> JSONResponse:
    """
    Retrieve a specific AI Touch Value™ calculation by ID.

    Args:
        calculation_id: MongoDB ObjectId of the calculation record
        current_user: Authenticated user from get_current_user dependency

    Returns:
        JSONResponse: PredictResponse with the calculation details

    Raises:
        HTTPException: 401 for unauthorized, 404 if not found, 500 for errors
    """
    # Extract user ID from authenticated user
    user_id = current_user.get("_id")
    if not user_id:
        logger.error("Authenticated user missing _id field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user authentication data",
        )

    logger.info(f"Fetching calculation {calculation_id} for user={user_id}")

    try:
        # Validate calculation_id format
        try:
            obj_id = ObjectId(calculation_id)
        except InvalidId:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid calculation ID format",
            ) from None

        # Get database and analytics collection
        database = get_database()
        analytics_collection = database["analytics"]

        # Find the calculation by ID and user_id (security check)
        record = await analytics_collection.find_one(
            {
                "_id": obj_id,
                "user_id": user_id,
            }
        )

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Calculation not found",
            )

        # Transform to response format
        calc = AITouchValueCalculation.from_mongodb_dict(record)

        calculated_value = float(calc.calculated_value) if calc.calculated_value else 0.0
        formula_breakdown = generate_formula_breakdown(
            model_earnings=float(calc.model_earnings),
            contribution_score=calc.training_contribution_score,
            exposure_score=calc.usage_exposure_score,
            calculated_value=calculated_value,
        )

        response_data = PredictResponse(
            calculated_value=calculated_value,
            equity_factor=calc.equity_factor,
            model_earnings=float(calc.model_earnings),
            training_contribution_score=calc.training_contribution_score,
            usage_exposure_score=calc.usage_exposure_score,
            calculation_id=calc.id,
            timestamp=calc.created_at.isoformat() if calc.created_at else None,
            formula_breakdown=formula_breakdown,
            asset_id=calc.asset_id,
            user_id=calc.user_id,
        )

        logger.info(f"Retrieved calculation {calculation_id} for user={user_id}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=response_data.model_dump(mode="json"),
        )

    except HTTPException:
        raise

    except RuntimeError:
        logger.exception("Database error fetching calculation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please try again later.",
        ) from None

    except Exception:
        logger.exception("Unexpected error fetching calculation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again.",
        ) from None
