"""
AI Touch Value™ Calculation Service for META-STAMP V3.

This service implements the core AI Touch Value™ calculation engine that determines
creator compensation based on their content's contribution to AI model training.

The calculation follows the exact formula specified in the Agent Action Plan:

    AI Touch Value™ = ModelEarnings * (TrainingContributionScore/100)
                      * (UsageExposureScore/100) * EquityFactor

Where:
    - ModelEarnings: Total earnings from the AI model in USD (≥ 0)
    - TrainingContributionScore: Score 0-100 representing data contribution
    - UsageExposureScore: Score 0-100 representing content exposure/usage
    - EquityFactor: Fixed at 0.25 (25%) - NON-NEGOTIABLE per requirements

The service provides:
    - Core calculation with strict input validation
    - Predictive modeling for contribution and exposure scores
    - Historical calculation tracking in MongoDB
    - Value projection over time periods
    - Transparent formula breakdown for audit purposes

Example:
    >>> service = AIValueService()
    >>> result = await service.calculate_ai_touch_value(
    ...     model_earnings=100000.00,
    ...     contribution_score=75.0,
    ...     exposure_score=60.0
    ... )
    >>> print(result['calculated_value'])
    11250.00

Notes:
    - All monetary values use Decimal for financial precision
    - Scores are validated to be within 0-100 range
    - The equity factor is immutable at 0.25 (25%)
"""

import logging
import math

from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.core.database import get_db_client
from app.models.analytics import AITouchValueCalculation


# Configure module logger for structured logging
logger = logging.getLogger(__name__)

# Maximum value for contribution and exposure scores (0-100 range)
MAX_SCORE = 100


class AIValueInput(BaseModel):
    """
    Pydantic model for validating AI Touch Value™ calculation inputs.

    This model enforces strict validation rules per Agent Action Plan section 0.8:
    - model_earnings must be non-negative (≥ 0)
    - training_contribution_score must be in range 0-100
    - usage_exposure_score must be in range 0-100

    Attributes:
        model_earnings: AI model earnings in USD, must be ≥ 0
        training_contribution_score: Training data contribution score 0-100
        usage_exposure_score: Usage/exposure score 0-100

    Example:
        >>> input_data = AIValueInput(
        ...     model_earnings=50000.0,
        ...     training_contribution_score=80.0,
        ...     usage_exposure_score=65.0
        ... )
        >>> print(input_data.model_earnings)
        50000.0
    """

    model_earnings: float = Field(
        ..., ge=0, description="Model earnings in dollars, must be non-negative"
    )
    training_contribution_score: float = Field(
        ..., ge=0, le=100, description="Training contribution score from 0-100"
    )
    usage_exposure_score: float = Field(
        ..., ge=0, le=100, description="Usage exposure score from 0-100"
    )

    @field_validator("model_earnings", mode="before")
    @classmethod
    def validate_earnings(cls, value: Any) -> float:
        """
        Validate that model_earnings is a non-negative number.

        Args:
            value: Input value to validate

        Returns:
            float: Validated model earnings value

        Raises:
            ValueError: If value is negative or not a valid number
        """
        if value is None:
            raise ValueError("Model earnings is required and cannot be None")

        try:
            earnings = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Model earnings must be a number, got {type(value).__name__}") from e

        if earnings < 0:
            raise ValueError("Model earnings must be non-negative")

        return earnings

    @field_validator("training_contribution_score", mode="before")
    @classmethod
    def validate_contribution_score(cls, value: Any) -> float:
        """
        Validate training_contribution_score is within 0-100 range.

        Args:
            value: Input score value

        Returns:
            float: Validated contribution score

        Raises:
            ValueError: If score is outside 0-100 range
        """
        if value is None:
            raise ValueError("Training contribution score is required")

        try:
            score = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Training contribution score must be a number, got {type(value).__name__}"
            ) from e

        if score < 0 or score > MAX_SCORE:
            raise ValueError(
                f"Training contribution score must be between 0 and {MAX_SCORE}, got {score}"
            )

        return score

    @field_validator("usage_exposure_score", mode="before")
    @classmethod
    def validate_exposure_score(cls, value: Any) -> float:
        """
        Validate usage_exposure_score is within 0-100 range.

        Args:
            value: Input score value

        Returns:
            float: Validated exposure score

        Raises:
            ValueError: If score is outside 0-100 range
        """
        if value is None:
            raise ValueError("Usage exposure score is required")

        try:
            score = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Usage exposure score must be a number, got {type(value).__name__}"
            ) from e

        if score < 0 or score > MAX_SCORE:
            raise ValueError(f"Usage exposure score must be between 0 and {MAX_SCORE}, got {score}")

        return score


class CreatorMetrics(BaseModel):
    """
    Pydantic model for creator metrics used in predictive scoring.

    This model captures key creator performance indicators used to predict
    contribution and exposure scores when actual scores are not available.

    Attributes:
        followers: Total follower count across platforms (≥ 0)
        content_hours: Total hours of content created (≥ 0)
        views: Total view count for creator's content (≥ 0)
        platform: Primary platform name (e.g., "youtube", "tiktok")

    Example:
        >>> metrics = CreatorMetrics(
        ...     followers=100000,
        ...     content_hours=500.5,
        ...     views=5000000,
        ...     platform="youtube"
        ... )
    """

    followers: int = Field(..., ge=0, description="Total follower count across platforms")
    content_hours: float = Field(..., ge=0, description="Total hours of content created")
    views: int = Field(..., ge=0, description="Total view count for creator's content")
    platform: str = Field(
        ..., min_length=1, description="Primary platform name (e.g., youtube, tiktok, instagram)"
    )

    @field_validator("platform", mode="before")
    @classmethod
    def validate_platform(cls, value: Any) -> str:
        """
        Validate and normalize platform name.

        Args:
            value: Platform name input

        Returns:
            str: Normalized lowercase platform name

        Raises:
            ValueError: If platform is empty
            TypeError: If platform is not a string
        """
        if value is None:
            raise ValueError("Platform is required")

        if not isinstance(value, str):
            raise TypeError(f"Platform must be a string, got {type(value).__name__}")

        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Platform cannot be empty")

        return normalized


class AIValueService:
    """
    AI Touch Value™ Calculation Service.

    This service provides the core calculation engine for determining creator
    compensation based on their content's contribution to AI model training.

    The calculation implements the exact formula from Agent Action Plan section 0.3:

        AI Touch Value™ = ModelEarnings * (TrainingContributionScore/100)
                          * (UsageExposureScore/100) * 0.25

    Key Features:
        - Strict input validation (model_earnings ≥ 0, scores 0-100)
        - Fixed equity factor at 0.25 (25%) - NON-NEGOTIABLE
        - Predictive scoring from creator metrics (MVP heuristic model)
        - Historical calculation tracking in MongoDB
        - Value projection over time periods
        - Transparent formula breakdown

    Attributes:
        EQUITY_FACTOR: Fixed equity factor at 0.25 (25%)
        logger: Module logger for operational tracking

    Example:
        >>> service = AIValueService()
        >>> result = await service.calculate_ai_touch_value(
        ...     model_earnings=100000.00,
        ...     contribution_score=75.0,
        ...     exposure_score=60.0,
        ...     user_id="user123",
        ...     asset_id="asset456"
        ... )
        >>> print(result['calculated_value'])
        '11250.00'
    """

    # Fixed equity factor at 0.25 (25%) - NON-NEGOTIABLE per Agent Action Plan
    EQUITY_FACTOR: float = 0.25

    def __init__(self) -> None:
        """
        Initialize the AIValueService.

        Sets up logging for operational tracking and calculation auditing.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AIValueService initialized with EQUITY_FACTOR={self.EQUITY_FACTOR}")

    async def calculate_ai_touch_value(
        self,
        model_earnings: float,
        contribution_score: float,
        exposure_score: float,
        user_id: str | None = None,
        asset_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Calculate AI Touch Value™ using the exact formula.

        Implements the mandatory formula from Agent Action Plan section 0.3:

            AI Touch Value™ = ModelEarnings * (TrainingContributionScore/100)
                              * (UsageExposureScore/100) * 0.25

        Args:
            model_earnings: AI model earnings in USD (must be ≥ 0)
            contribution_score: Training contribution score (0-100)
            exposure_score: Usage exposure score (0-100)
            user_id: Optional user identifier for record association
            asset_id: Optional asset identifier for asset-specific calculations

        Returns:
            Dict containing:
                - calculated_value: The computed AI Touch Value™
                - model_earnings: Input model earnings
                - contribution_score: Input contribution score
                - exposure_score: Input exposure score
                - equity_factor: The fixed 0.25 equity factor
                - calculation_breakdown: Step-by-step formula breakdown
                - timestamp: ISO format UTC timestamp

        Raises:
            ValueError: If inputs fail validation

        Example:
            >>> result = await service.calculate_ai_touch_value(
            ...     model_earnings=100000.00,
            ...     contribution_score=75.0,
            ...     exposure_score=60.0
            ... )
            >>> result['calculated_value']
            '11250.00'
        """
        self.logger.info(
            f"Calculating AI Touch Value™: earnings=${model_earnings}, "
            f"contribution={contribution_score}, exposure={exposure_score}"
        )

        # Validate inputs using Pydantic model
        try:
            validated_input = AIValueInput(
                model_earnings=model_earnings,
                training_contribution_score=contribution_score,
                usage_exposure_score=exposure_score,
            )
        except ValueError:
            self.logger.exception("Input validation failed")
            raise

        # Calculate using Decimal for financial precision
        earnings_decimal = Decimal(str(validated_input.model_earnings))
        contribution_factor = Decimal(str(validated_input.training_contribution_score)) / Decimal(
            "100"
        )
        exposure_factor = Decimal(str(validated_input.usage_exposure_score)) / Decimal("100")
        equity_decimal = Decimal(str(self.EQUITY_FACTOR))

        # Apply the exact formula
        calculated_value = earnings_decimal * contribution_factor * exposure_factor * equity_decimal

        # Round to 2 decimal places for currency precision
        calculated_value = calculated_value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Get formula breakdown
        breakdown = self.get_formula_breakdown(
            model_earnings=validated_input.model_earnings,
            contribution_score=validated_input.training_contribution_score,
            exposure_score=validated_input.usage_exposure_score,
        )

        # Create timestamp
        timestamp = datetime.now(UTC)

        # Store in MongoDB if user_id is provided
        calculation_id = None
        if user_id:
            try:
                calculation_record = AITouchValueCalculation(
                    user_id=user_id,
                    asset_id=asset_id,
                    model_earnings=earnings_decimal,
                    training_contribution_score=validated_input.training_contribution_score,
                    usage_exposure_score=validated_input.usage_exposure_score,
                    equity_factor=self.EQUITY_FACTOR,
                    calculated_value=calculated_value,
                    created_at=timestamp,
                )

                # Get database client and store record
                db_client = get_db_client()
                analytics_collection = db_client.get_analytics_collection()
                insert_result = await analytics_collection.insert_one(
                    calculation_record.to_mongodb_dict()
                )
                calculation_id = str(insert_result.inserted_id)

                self.logger.info(f"Stored calculation record with ID: {calculation_id}")
            except RuntimeError as e:
                # Database not initialized - log warning but continue
                self.logger.warning(f"Could not store calculation in database: {e}")
            except Exception:
                self.logger.exception("Failed to store calculation in database")

        result = {
            "calculated_value": str(calculated_value),
            "model_earnings": str(earnings_decimal),
            "contribution_score": validated_input.training_contribution_score,
            "exposure_score": validated_input.usage_exposure_score,
            "equity_factor": self.EQUITY_FACTOR,
            "calculation_breakdown": breakdown,
            "timestamp": timestamp.isoformat(),
            "calculation_id": calculation_id,
            "user_id": user_id,
            "asset_id": asset_id,
        }

        self.logger.info(f"AI Touch Value™ calculated: ${calculated_value}")

        return result

    async def predict_contribution_score(self, creator_metrics: CreatorMetrics) -> float:
        """
        Predict training contribution score from creator metrics.

        This method uses a heuristic model to estimate how much a creator's
        content has likely contributed to AI model training datasets. The
        score considers content volume and platform factors.

        Note: This is a simplified heuristic model for MVP. Phase 2 will
        implement ML-based prediction using actual dataset analysis.

        Args:
            creator_metrics: Creator performance metrics

        Returns:
            float: Predicted contribution score (0-100)

        Prediction Factors:
            - content_hours: More content = higher training contribution
            - platform: Some platforms more likely in training data
        """
        self.logger.info(f"Predicting contribution score for platform={creator_metrics.platform}")

        base_score = 0.0

        # Content volume factor (up to 50 points)
        # Scale: 0-1000 hours maps to 0-50 points
        content_factor = min(creator_metrics.content_hours / 1000.0, 1.0) * 50.0
        base_score += content_factor

        # Platform factor (up to 30 points)
        # Different platforms have different likelihoods of being in training data
        platform_scores = {
            "youtube": 30.0,  # High: Major source for video/text training
            "twitter": 28.0,  # High: Common text training source
            "x": 28.0,  # Same as Twitter
            "reddit": 27.0,  # High: Common in text datasets
            "github": 30.0,  # High: Major code training source
            "wikipedia": 30.0,  # High: Standard training data
            "medium": 25.0,  # Medium-High: Blog content
            "instagram": 20.0,  # Medium: Image/caption training
            "tiktok": 22.0,  # Medium: Growing video source
            "vimeo": 18.0,  # Medium: Video content
            "spotify": 15.0,  # Lower: Audio content
            "soundcloud": 14.0,  # Lower: Audio content
            "twitch": 16.0,  # Lower: Streaming content
            "pinterest": 12.0,  # Lower: Image curation
            "linkedin": 15.0,  # Medium-Low: Professional content
        }
        platform_score = platform_scores.get(creator_metrics.platform, 15.0)
        base_score += platform_score

        # Engagement factor based on content ratio (up to 20 points)
        # Higher content hours per follower suggests dedicated creator
        if creator_metrics.followers > 0:
            hours_per_1k_followers = creator_metrics.content_hours / (
                creator_metrics.followers / 1000.0
            )
            engagement_factor = min(hours_per_1k_followers / 10.0, 1.0) * 20.0
            base_score += engagement_factor
        else:
            # If no followers, assume moderate engagement
            base_score += 10.0

        # Ensure score is within 0-100 range
        final_score = max(0.0, min(100.0, base_score))

        self.logger.info(f"Predicted contribution score: {final_score:.2f}")

        return round(final_score, 2)

    async def predict_exposure_score(self, creator_metrics: CreatorMetrics) -> float:
        """
        Predict usage exposure score from creator metrics.

        This method estimates how much a creator's content is likely being
        exposed and used based on their reach and engagement metrics.

        Note: This is a simplified heuristic model for MVP. Phase 2 will
        implement ML-based prediction using actual usage tracking.

        Args:
            creator_metrics: Creator performance metrics

        Returns:
            float: Predicted exposure score (0-100)

        Prediction Factors:
            - followers: Higher follower count = more exposure
            - views: Higher views = more content consumption
            - platform: Platform popularity affects exposure
        """
        self.logger.info(f"Predicting exposure score for platform={creator_metrics.platform}")

        base_score = 0.0

        # Follower reach factor (up to 40 points)
        # Scale: 0-1M followers maps to 0-40 points (logarithmic)
        if creator_metrics.followers > 0:
            # Use log scale: 1K followers = 10 points, 1M = 40 points
            follower_factor = (math.log10(max(creator_metrics.followers, 1)) / 6.0) * 40.0
            follower_factor = min(follower_factor, 40.0)
            base_score += follower_factor
        else:
            base_score += 0.0

        # View engagement factor (up to 40 points)
        # Scale: 0-10M views maps to 0-40 points (logarithmic)
        if creator_metrics.views > 0:
            # Use log scale: 10K views = 10 points, 10M = 40 points
            view_factor = (math.log10(max(creator_metrics.views, 1)) / 7.0) * 40.0
            view_factor = min(view_factor, 40.0)
            base_score += view_factor
        else:
            base_score += 0.0

        # Platform popularity factor (up to 20 points)
        # More popular platforms have higher exposure potential
        platform_exposure = {
            "youtube": 20.0,  # Highest: Massive global reach
            "tiktok": 19.0,  # Very High: Viral potential
            "instagram": 18.0,  # High: Visual platform
            "twitter": 17.0,  # High: News/viral content
            "x": 17.0,  # Same as Twitter
            "facebook": 16.0,  # High: Large user base
            "reddit": 15.0,  # Medium-High: Engaged communities
            "spotify": 14.0,  # Medium: Audio platform
            "twitch": 14.0,  # Medium: Live streaming
            "linkedin": 12.0,  # Medium: Professional network
            "github": 13.0,  # Medium: Developer community
            "medium": 11.0,  # Medium: Blog platform
            "vimeo": 10.0,  # Lower: Niche video
            "soundcloud": 9.0,  # Lower: Audio niche
            "pinterest": 10.0,  # Medium-Low: Visual discovery
        }
        platform_score = platform_exposure.get(creator_metrics.platform, 10.0)
        base_score += platform_score

        # Ensure score is within 0-100 range
        final_score = max(0.0, min(100.0, base_score))

        self.logger.info(f"Predicted exposure score: {final_score:.2f}")

        return round(final_score, 2)

    async def predict_value_from_metrics(
        self,
        model_earnings: float,
        creator_metrics: CreatorMetrics,
        user_id: str | None = None,
        asset_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Predict AI Touch Value™ from creator metrics.

        This method combines score prediction with value calculation to provide
        a complete AI Touch Value™ estimate based on creator metrics.

        Args:
            model_earnings: AI model earnings in USD
            creator_metrics: Creator performance metrics
            user_id: Optional user identifier for record storage
            asset_id: Optional asset identifier

        Returns:
            Dict containing:
                - predicted_contribution_score: Estimated contribution score
                - predicted_exposure_score: Estimated exposure score
                - calculated_value: Computed AI Touch Value™
                - creator_metrics: Input metrics used
                - calculation_details: Full calculation breakdown

        Example:
            >>> metrics = CreatorMetrics(
            ...     followers=100000,
            ...     content_hours=500.0,
            ...     views=5000000,
            ...     platform="youtube"
            ... )
            >>> result = await service.predict_value_from_metrics(
            ...     model_earnings=100000.00,
            ...     creator_metrics=metrics
            ... )
        """
        self.logger.info(
            f"Predicting AI Touch Value™ from metrics for platform={creator_metrics.platform}"
        )

        # Predict both scores from creator metrics
        contribution_score = await self.predict_contribution_score(creator_metrics)
        exposure_score = await self.predict_exposure_score(creator_metrics)

        # Calculate AI Touch Value using predicted scores
        calculation_result = await self.calculate_ai_touch_value(
            model_earnings=model_earnings,
            contribution_score=contribution_score,
            exposure_score=exposure_score,
            user_id=user_id,
            asset_id=asset_id,
        )

        result = {
            "predicted_contribution_score": contribution_score,
            "predicted_exposure_score": exposure_score,
            "calculated_value": calculation_result["calculated_value"],
            "model_earnings": calculation_result["model_earnings"],
            "equity_factor": self.EQUITY_FACTOR,
            "creator_metrics": {
                "followers": creator_metrics.followers,
                "content_hours": creator_metrics.content_hours,
                "views": creator_metrics.views,
                "platform": creator_metrics.platform,
            },
            "calculation_details": calculation_result,
            "prediction_method": "heuristic_v1",
            "timestamp": calculation_result["timestamp"],
        }

        self.logger.info(f"Predicted AI Touch Value™: ${calculation_result['calculated_value']}")

        return result

    async def get_calculation_history(
        self,
        asset_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve historical AI Touch Value™ calculations for an asset.

        Queries MongoDB for past calculations associated with the specified
        asset, enabling trend analysis and historical tracking.

        Args:
            asset_id: Asset identifier to query calculations for
            limit: Maximum number of records to return (default: 10)

        Returns:
            List of calculation records with all fields including:
                - calculated_value
                - model_earnings
                - contribution_score
                - exposure_score
                - timestamp

        Raises:
            RuntimeError: If database is not initialized

        Example:
            >>> history = await service.get_calculation_history(
            ...     asset_id="asset123",
            ...     limit=5
            ... )
            >>> len(history)
            5
        """
        self.logger.info(f"Retrieving calculation history for asset_id={asset_id}, limit={limit}")

        try:
            db_client = get_db_client()
            analytics_collection = db_client.get_analytics_collection()

            # Query calculations sorted by created_at descending
            cursor = (
                analytics_collection.find({"asset_id": asset_id})
                .sort("created_at", -1)
                .limit(limit)
            )

            records = []
            async for doc in cursor:
                # Convert MongoDB document to dictionary
                record = AITouchValueCalculation.from_mongodb_dict(doc)
                records.append(
                    {
                        "id": record.id,
                        "user_id": record.user_id,
                        "asset_id": record.asset_id,
                        "model_earnings": str(record.model_earnings),
                        "contribution_score": record.training_contribution_score,
                        "exposure_score": record.usage_exposure_score,
                        "calculated_value": str(record.calculated_value),
                        "equity_factor": record.equity_factor,
                        "timestamp": record.created_at.isoformat(),
                    }
                )

            self.logger.info(
                f"Retrieved {len(records)} calculation records for asset_id={asset_id}"
            )

            return records

        except RuntimeError:
            self.logger.exception("Database error retrieving calculation history")
            raise
        except Exception:
            self.logger.exception("Error retrieving calculation history")
            raise

    async def project_value(
        self,
        base_model_earnings: float,
        contribution_score: float,
        exposure_score: float,
        growth_rate: float = 0.1,
        periods: int = 12,
    ) -> list[dict[str, Any]]:
        """
        Project AI Touch Value™ over future time periods.

        Calculates projected values assuming model earnings grow at the
        specified growth rate. Useful for creator earnings forecasting.

        Args:
            base_model_earnings: Starting model earnings in USD
            contribution_score: Training contribution score (0-100)
            exposure_score: Usage exposure score (0-100)
            growth_rate: Expected period-over-period earnings growth (default: 10%)
            periods: Number of periods to project (default: 12 months)

        Returns:
            List of projected values by period containing:
                - period: Period number (1-indexed)
                - projected_earnings: Model earnings for this period
                - projected_value: AI Touch Value™ for this period
                - cumulative_value: Running total of all periods

        Example:
            >>> projections = await service.project_value(
            ...     base_model_earnings=100000.00,
            ...     contribution_score=75.0,
            ...     exposure_score=60.0,
            ...     growth_rate=0.1,
            ...     periods=12
            ... )
            >>> len(projections)
            12
        """
        self.logger.info(
            f"Projecting AI Touch Value™ over {periods} periods "
            f"with {growth_rate*100}% growth rate"
        )

        # Validate inputs
        validated_input = AIValueInput(
            model_earnings=base_model_earnings,
            training_contribution_score=contribution_score,
            usage_exposure_score=exposure_score,
        )

        projections = []
        cumulative_value = Decimal("0.00")
        current_earnings = Decimal(str(validated_input.model_earnings))

        # Calculate factors once for efficiency
        contribution_factor = Decimal(str(validated_input.training_contribution_score)) / Decimal(
            "100"
        )
        exposure_factor = Decimal(str(validated_input.usage_exposure_score)) / Decimal("100")
        equity_decimal = Decimal(str(self.EQUITY_FACTOR))
        growth_decimal = Decimal(str(1 + growth_rate))

        for period in range(1, periods + 1):
            # Calculate value for this period
            period_value = (
                current_earnings * contribution_factor * exposure_factor * equity_decimal
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            cumulative_value += period_value

            projections.append(
                {
                    "period": period,
                    "projected_earnings": str(
                        current_earnings.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    ),
                    "projected_value": str(period_value),
                    "cumulative_value": str(
                        cumulative_value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    ),
                    "contribution_score": validated_input.training_contribution_score,
                    "exposure_score": validated_input.usage_exposure_score,
                    "equity_factor": self.EQUITY_FACTOR,
                }
            )

            # Apply growth for next period
            current_earnings = (current_earnings * growth_decimal).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        self.logger.info(
            f"Generated {len(projections)} period projections, "
            f"cumulative value: ${cumulative_value}"
        )

        return projections

    def get_formula_breakdown(
        self,
        model_earnings: float,
        contribution_score: float,
        exposure_score: float,
    ) -> dict[str, Any]:
        """
        Get detailed breakdown of AI Touch Value™ calculation.

        Provides step-by-step calculation transparency for audit purposes
        and creator understanding of how their compensation is determined.

        Args:
            model_earnings: AI model earnings in USD
            contribution_score: Training contribution score (0-100)
            exposure_score: Usage exposure score (0-100)

        Returns:
            Dict containing:
                - formula: String representation of the formula
                - inputs: All input values
                - intermediate_values: Step-by-step calculations
                - final_result: The computed AI Touch Value™

        Example:
            >>> breakdown = service.get_formula_breakdown(
            ...     model_earnings=100000.00,
            ...     contribution_score=75.0,
            ...     exposure_score=60.0
            ... )
            >>> breakdown['formula']
            'AI Touch Value™ = ModelEarnings * (ContributionScore/100) * (ExposureScore/100) * EquityFactor'
        """
        # Convert to Decimal for precise calculations
        earnings_decimal = Decimal(str(model_earnings))
        contribution_factor = Decimal(str(contribution_score)) / Decimal("100")
        exposure_factor = Decimal(str(exposure_score)) / Decimal("100")
        equity_decimal = Decimal(str(self.EQUITY_FACTOR))

        # Calculate intermediate values
        step1 = earnings_decimal * contribution_factor
        step2 = step1 * exposure_factor
        final_value = (step2 * equity_decimal).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Calculate combined multiplier
        combined_multiplier = (contribution_factor * exposure_factor * equity_decimal).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )

        return {
            "formula": (
                "AI Touch Value™ = ModelEarnings * (ContributionScore/100) "
                "* (ExposureScore/100) * EquityFactor"
            ),
            "inputs": {
                "model_earnings": str(earnings_decimal),
                "contribution_score": contribution_score,
                "exposure_score": exposure_score,
                "equity_factor": self.EQUITY_FACTOR,
            },
            "intermediate_values": {
                "step1_earnings_x_contribution": str(
                    step1.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                ),
                "step1_description": f"${model_earnings} * ({contribution_score}/100)",
                "step2_x_exposure": str(step2.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
                "step2_description": f"Step1 * ({exposure_score}/100)",
                "step3_x_equity": str(final_value),
                "step3_description": f"Step2 * {self.EQUITY_FACTOR}",
            },
            "factors": {
                "contribution_factor": float(contribution_factor),
                "exposure_factor": float(exposure_factor),
                "equity_factor": self.EQUITY_FACTOR,
                "combined_multiplier": float(combined_multiplier),
            },
            "final_result": {
                "calculated_value": str(final_value),
                "currency": "USD",
                "explanation": (
                    f"The creator is entitled to ${final_value} based on "
                    f"their {contribution_score}% training contribution, "
                    f"{exposure_score}% usage exposure, and the fixed "
                    f"{self.EQUITY_FACTOR * 100}% equity factor."
                ),
            },
        }


# Export all public classes
__all__ = ["AIValueInput", "AIValueService", "CreatorMetrics"]
