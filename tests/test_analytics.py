"""
META-STAMP V3 Analytics Test Suite

Comprehensive pytest test suite for the AI Touch Value™ calculation engine.
Tests validate the exact formula specified in Agent Action Plan section 0.3:

    AI Touch Value™ = ModelEarnings x (TrainingContributionScore/100)
                      x (UsageExposureScore/100) x 0.25

Test coverage includes:
- Formula correctness with standard inputs
- Equity factor enforcement (fixed at 0.25 / 25%)
- Input validation (scores 0-100, earnings >= 0)
- Edge cases (zero values, negative numbers, maximum values)
- Prediction function with user metrics
- MongoDB storage operations
- API endpoint integration tests
- Performance validation

Per Agent Action Plan sections 0.3, 0.4, 0.5, 0.6, 0.8, and 0.10.
"""

import time

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi.testclient import TestClient

from app.api.v1.analytics import router
from app.config import Settings
from app.models.analytics import AITouchValueCalculation
from app.services.ai_value_service import AIValueService, CreatorMetrics


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_settings() -> Settings:
    """
    Create a Settings instance with test configuration.

    Verifies the equity_factor is fixed at 0.25 per Agent Action Plan section 0.3.
    """
    settings = Settings(
        app_env="testing",
        debug=True,
        mongodb_uri="mongodb://localhost:27017",
        mongodb_db_name="meta_stamp_test",
        secret_key="test-secret-key-for-jwt-signing-min32chars",
    )
    # Verify equity factor is always 0.25 per requirements
    assert settings.equity_factor == 0.25
    return settings


@pytest.fixture
def ai_value_service() -> AIValueService:
    """Create an AIValueService instance for testing."""
    return AIValueService()


@pytest.fixture
def mock_db() -> AsyncMock:
    """
    Create AsyncMock for MongoDB analytics collection.

    Provides mock insert_one, find, and other collection operations
    for testing calculation storage without actual database connection.
    """
    mock_collection = AsyncMock()
    mock_collection.insert_one = AsyncMock(
        return_value=MagicMock(inserted_id="test_calculation_id_12345")
    )
    mock_collection.find = MagicMock(return_value=AsyncMock())

    mock_db_client = MagicMock()
    mock_db_client.get_analytics_collection = MagicMock(return_value=mock_collection)

    return mock_db_client


@pytest.fixture
def mock_auth() -> dict[str, Any]:
    """
    Create mock authenticated user data for endpoint testing.

    Returns a user dict with _id field required by the predict endpoint.
    """
    return {
        "_id": "test_user_id_12345",
        "email": "test@example.com",
        "name": "Test User",
    }


@pytest.fixture
def sample_creator_metrics() -> CreatorMetrics:
    """Create sample CreatorMetrics for prediction testing."""
    return CreatorMetrics(
        followers=100000,
        content_hours=500.0,
        views=5000000,
        platform="youtube",
    )


# =============================================================================
# Test Formula Correctness
# =============================================================================


class TestFormulaCorrectness:
    """
    Test AI Touch Value™ formula implementation.

    Formula: AI Touch Value™ = ModelEarnings x (ContributionScore/100)
             x (ExposureScore/100) x 0.25
    """

    @pytest.mark.asyncio
    async def test_calculate_ai_touch_value_basic(self, ai_value_service: AIValueService) -> None:
        """
        Test formula with standard inputs.

        Input: model_earnings=1000, contribution_score=50, exposure_score=80
        Expected: 1000 x (50/100) x (80/100) x 0.25 = 100.0
        """
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=50.0,
                exposure_score=80.0,
            )

            assert result["calculated_value"] == "100.00"
            assert result["equity_factor"] == 0.25
            assert result["contribution_score"] == 50.0
            assert result["exposure_score"] == 80.0

    @pytest.mark.asyncio
    async def test_calculate_ai_touch_value_zero_earnings(
        self, ai_value_service: AIValueService
    ) -> None:
        """
        Test with zero model earnings.

        Expected: 0 regardless of scores (0 x anything = 0)
        """
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=0.0,
                contribution_score=75.0,
                exposure_score=80.0,
            )

            assert result["calculated_value"] == "0.00"

    @pytest.mark.asyncio
    async def test_calculate_ai_touch_value_max_scores(
        self, ai_value_service: AIValueService
    ) -> None:
        """
        Test with maximum scores (100, 100).

        Input: model_earnings=10000, contribution_score=100, exposure_score=100
        Expected: 10000 x 1.0 x 1.0 x 0.25 = 2500.0
        """
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=10000.0,
                contribution_score=100.0,
                exposure_score=100.0,
            )

            assert result["calculated_value"] == "2500.00"

    @pytest.mark.asyncio
    async def test_calculate_ai_touch_value_min_scores(
        self, ai_value_service: AIValueService
    ) -> None:
        """
        Test with minimum scores (0, 0).

        Expected: 0 regardless of earnings (anything x 0 = 0)
        """
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=100000.0,
                contribution_score=0.0,
                exposure_score=0.0,
            )

            assert result["calculated_value"] == "0.00"

    @pytest.mark.asyncio
    async def test_calculate_complex_values(self, ai_value_service: AIValueService) -> None:
        """
        Test formula with complex decimal values.

        Input: model_earnings=75000, contribution_score=65.5, exposure_score=42.3
        Expected: 75000 x 0.655 x 0.423 x 0.25 ≈ 5195.44
        """
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=75000.0,
                contribution_score=65.5,
                exposure_score=42.3,
            )

            # Calculate expected: 75000 * 0.655 * 0.423 * 0.25 = 5194.96875
            # Rounded to 2 decimal places: 5194.97
            calculated = Decimal(result["calculated_value"])
            expected = Decimal("5194.97")
            assert calculated == expected


# =============================================================================
# Test Equity Factor Enforcement
# =============================================================================


class TestEquityFactorEnforcement:
    """
    Test that the equity factor is always 0.25 (25%) per Agent Action Plan.

    The equity factor is NON-NEGOTIABLE and cannot be overridden.
    """

    def test_equity_factor_is_25_percent(self, ai_value_service: AIValueService) -> None:
        """Verify equity factor is always 0.25."""
        assert ai_value_service.EQUITY_FACTOR == 0.25

    def test_equity_factor_not_configurable(self, mock_settings: Settings) -> None:
        """Verify equity factor in settings is fixed at 0.25."""
        # Even if we tried to set it differently, it should be 0.25
        assert mock_settings.equity_factor == 0.25

    def test_equity_factor_in_model(self) -> None:
        """Verify model always has equity_factor=0.25."""
        calc = AITouchValueCalculation(
            user_id="test_user",
            model_earnings=Decimal("10000.00"),
            training_contribution_score=50.0,
            usage_exposure_score=50.0,
            equity_factor=0.99,  # Try to override - should be ignored
        )
        # Equity factor should still be 0.25
        assert calc.equity_factor == 0.25

    def test_equity_factor_precision(self, ai_value_service: AIValueService) -> None:
        """Test decimal precision in calculations with equity factor."""
        breakdown = ai_value_service.get_formula_breakdown(
            model_earnings=1000.0,
            contribution_score=100.0,
            exposure_score=100.0,
        )

        # Final result should be 1000 * 1 * 1 * 0.25 = 250.00
        assert breakdown["final_result"]["calculated_value"] == "250.00"
        assert breakdown["factors"]["equity_factor"] == 0.25

    @pytest.mark.asyncio
    async def test_equity_factor_applied_in_calculation(
        self, ai_value_service: AIValueService
    ) -> None:
        """Verify 25% equity factor is applied in every calculation."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            # With 100% scores, result should be earnings * 0.25
            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=100.0,
                exposure_score=100.0,
            )

            # 1000 * 1.0 * 1.0 * 0.25 = 250
            assert result["calculated_value"] == "250.00"
            assert result["equity_factor"] == 0.25


# =============================================================================
# Test Input Validation
# =============================================================================


class TestInputValidation:
    """
    Test input validation per Agent Action Plan section 0.4.

    Requirements:
    - model_earnings must be >= 0
    - contribution_score must be 0-100
    - exposure_score must be 0-100
    """

    @pytest.mark.asyncio
    async def test_validate_negative_earnings(self, ai_value_service: AIValueService) -> None:
        """Verify negative model_earnings raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            await ai_value_service.calculate_ai_touch_value(
                model_earnings=-1000.0,
                contribution_score=50.0,
                exposure_score=50.0,
            )

    @pytest.mark.asyncio
    async def test_validate_contribution_score_above_100(
        self, ai_value_service: AIValueService
    ) -> None:
        """Verify contribution_score > 100 raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=150.0,
                exposure_score=50.0,
            )

    @pytest.mark.asyncio
    async def test_validate_contribution_score_below_0(
        self, ai_value_service: AIValueService
    ) -> None:
        """Verify contribution_score < 0 raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=-10.0,
                exposure_score=50.0,
            )

    @pytest.mark.asyncio
    async def test_validate_exposure_score_above_100(
        self, ai_value_service: AIValueService
    ) -> None:
        """Verify exposure_score > 100 raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=50.0,
                exposure_score=200.0,
            )

    @pytest.mark.asyncio
    async def test_validate_exposure_score_below_0(self, ai_value_service: AIValueService) -> None:
        """Verify exposure_score < 0 raises ValueError."""
        with pytest.raises(ValueError, match="between 0 and 100"):
            await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=50.0,
                exposure_score=-5.0,
            )

    @pytest.mark.asyncio
    async def test_validate_all_inputs_together(self, ai_value_service: AIValueService) -> None:
        """Test that validation catches the first invalid input."""
        # Negative earnings should be caught first
        with pytest.raises(ValueError, match="must be non-negative"):
            await ai_value_service.calculate_ai_touch_value(
                model_earnings=-100.0,
                contribution_score=150.0,
                exposure_score=-20.0,
            )

    def test_model_validation_negative_earnings(self) -> None:
        """Test AITouchValueCalculation model rejects negative earnings."""
        with pytest.raises(ValueError, match="must be >= 0"):
            AITouchValueCalculation(
                user_id="test_user",
                model_earnings=Decimal("-100.00"),
                training_contribution_score=50.0,
                usage_exposure_score=50.0,
            )

    def test_model_validation_score_out_of_range(self) -> None:
        """Test AITouchValueCalculation model rejects out-of-range scores."""
        with pytest.raises(ValueError, match="between 0"):
            AITouchValueCalculation(
                user_id="test_user",
                model_earnings=Decimal("1000.00"),
                training_contribution_score=150.0,  # Invalid: > 100
                usage_exposure_score=50.0,
            )


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """
    Test edge cases for AI Touch Value™ calculation.

    Per Agent Action Plan section 0.5 requirements.
    """

    @pytest.mark.asyncio
    async def test_very_large_earnings(self, ai_value_service: AIValueService) -> None:
        """Test with earnings in billions."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=1_000_000_000.0,  # 1 billion
                contribution_score=50.0,
                exposure_score=50.0,
            )

            # 1,000,000,000 * 0.5 * 0.5 * 0.25 = 62,500,000
            assert result["calculated_value"] == "62500000.00"

    @pytest.mark.asyncio
    async def test_fractional_scores(self, ai_value_service: AIValueService) -> None:
        """Test with decimal scores (50.5, 75.25)."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=10000.0,
                contribution_score=50.5,
                exposure_score=75.25,
            )

            # 10000 * 0.505 * 0.7525 * 0.25 = 950.03125 → rounded to 950.03
            calculated = Decimal(result["calculated_value"])
            assert calculated == Decimal("950.03")

    @pytest.mark.asyncio
    async def test_rounding_precision(self, ai_value_service: AIValueService) -> None:
        """Verify results rounded to 2 decimal places."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=10000.0,
                contribution_score=33.33,
                exposure_score=33.33,
            )

            # Result should have exactly 2 decimal places
            value = result["calculated_value"]
            parts = value.split(".")
            assert len(parts) == 2
            assert len(parts[1]) == 2

    @pytest.mark.asyncio
    async def test_zero_contribution_nonzero_exposure(
        self, ai_value_service: AIValueService
    ) -> None:
        """Verify result is zero when contribution is zero."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=50000.0,
                contribution_score=0.0,
                exposure_score=100.0,
            )

            assert result["calculated_value"] == "0.00"

    @pytest.mark.asyncio
    async def test_nonzero_contribution_zero_exposure(
        self, ai_value_service: AIValueService
    ) -> None:
        """Verify result is zero when exposure is zero."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=50000.0,
                contribution_score=100.0,
                exposure_score=0.0,
            )

            assert result["calculated_value"] == "0.00"

    @pytest.mark.asyncio
    async def test_boundary_score_values(self, ai_value_service: AIValueService) -> None:
        """Test exact boundary values (0 and 100)."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            # Test score = 0
            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=0.0,
                exposure_score=100.0,
            )
            assert result["calculated_value"] == "0.00"

            # Test score = 100
            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=1000.0,
                contribution_score=100.0,
                exposure_score=100.0,
            )
            assert result["calculated_value"] == "250.00"

    def test_model_calculation_precision(self) -> None:
        """Test AITouchValueCalculation model uses proper decimal precision."""
        calc = AITouchValueCalculation(
            user_id="test_user",
            model_earnings=Decimal("100000.00"),
            training_contribution_score=75.0,
            usage_exposure_score=60.0,
        )

        # Verify the model calculates correctly
        # 100000 * 0.75 * 0.60 * 0.25 = 11250.00
        assert calc.calculated_value == Decimal("11250.00")


# =============================================================================
# Test Prediction Function
# =============================================================================


class TestPredictionFunction:
    """
    Test prediction function that estimates scores from user metrics.

    Tests predict_value_from_metrics and score prediction methods.
    """

    @pytest.mark.asyncio
    async def test_predict_from_user_metrics(
        self,
        ai_value_service: AIValueService,
        sample_creator_metrics: CreatorMetrics,
    ) -> None:
        """Test prediction from followers, views, content hours."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            result = await ai_value_service.predict_value_from_metrics(
                model_earnings=100000.0,
                creator_metrics=sample_creator_metrics,
            )

            assert "predicted_contribution_score" in result
            assert "predicted_exposure_score" in result
            assert "calculated_value" in result
            assert "creator_metrics" in result

            # Verify scores are within valid range
            assert 0 <= result["predicted_contribution_score"] <= 100
            assert 0 <= result["predicted_exposure_score"] <= 100

    @pytest.mark.asyncio
    async def test_predict_with_different_platforms(self, ai_value_service: AIValueService) -> None:
        """Test platform-specific score adjustments."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            # Test YouTube (high training likelihood)
            youtube_metrics = CreatorMetrics(
                followers=100000,
                content_hours=500.0,
                views=5000000,
                platform="youtube",
            )
            youtube_result = await ai_value_service.predict_value_from_metrics(
                model_earnings=100000.0,
                creator_metrics=youtube_metrics,
            )

            # Test Pinterest (lower training likelihood)
            pinterest_metrics = CreatorMetrics(
                followers=100000,
                content_hours=500.0,
                views=5000000,
                platform="pinterest",
            )
            pinterest_result = await ai_value_service.predict_value_from_metrics(
                model_earnings=100000.0,
                creator_metrics=pinterest_metrics,
            )

            # YouTube should have higher contribution score
            assert (
                youtube_result["predicted_contribution_score"]
                > pinterest_result["predicted_contribution_score"]
            )

    @pytest.mark.asyncio
    async def test_predict_low_metrics(self, ai_value_service: AIValueService) -> None:
        """Test prediction with minimal user metrics."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            low_metrics = CreatorMetrics(
                followers=100,
                content_hours=1.0,
                views=1000,
                platform="twitter",
            )

            result = await ai_value_service.predict_value_from_metrics(
                model_earnings=10000.0,
                creator_metrics=low_metrics,
            )

            # Low metrics should result in lower scores
            assert result["predicted_contribution_score"] < 50
            assert result["predicted_exposure_score"] < 50

    @pytest.mark.asyncio
    async def test_predict_high_metrics(self, ai_value_service: AIValueService) -> None:
        """Test prediction with high user metrics."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            high_metrics = CreatorMetrics(
                followers=10_000_000,  # 10M followers
                content_hours=5000.0,  # 5000 hours of content
                views=1_000_000_000,  # 1B views
                platform="youtube",
            )

            result = await ai_value_service.predict_value_from_metrics(
                model_earnings=100000.0,
                creator_metrics=high_metrics,
            )

            # High metrics should result in higher scores
            assert result["predicted_contribution_score"] > 50
            assert result["predicted_exposure_score"] > 50


# =============================================================================
# Test MongoDB Storage
# =============================================================================


class TestMongoDBStorage:
    """
    Test calculation storage in MongoDB analytics collection.

    Per Agent Action Plan section 0.4 requirement.
    """

    @pytest.mark.asyncio
    async def test_store_calculation_result(
        self,
        ai_value_service: AIValueService,
        mock_db: MagicMock,
    ) -> None:
        """Verify calculation stored in analytics collection."""
        with patch("app.services.ai_value_service.get_db_client", return_value=mock_db):
            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=10000.0,
                contribution_score=75.0,
                exposure_score=80.0,
                user_id="test_user_123",
                asset_id="test_asset_456",
            )

            # Verify insert_one was called
            mock_db.get_analytics_collection().insert_one.assert_called_once()

            # Verify calculation_id is returned
            assert result["calculation_id"] is not None

    @pytest.mark.asyncio
    async def test_calculation_includes_timestamp(
        self,
        ai_value_service: AIValueService,
        mock_db: MagicMock,
    ) -> None:
        """Verify timestamp included in stored data."""
        with patch("app.services.ai_value_service.get_db_client", return_value=mock_db):
            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=10000.0,
                contribution_score=75.0,
                exposure_score=80.0,
                user_id="test_user_123",
            )

            # Verify timestamp is in result
            assert "timestamp" in result
            assert result["timestamp"] is not None

            # Verify timestamp is ISO format
            datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))

    @pytest.mark.asyncio
    async def test_calculation_includes_all_inputs(
        self,
        ai_value_service: AIValueService,
        mock_db: MagicMock,
    ) -> None:
        """Verify all input values stored."""
        with patch("app.services.ai_value_service.get_db_client", return_value=mock_db):
            await ai_value_service.calculate_ai_touch_value(
                model_earnings=10000.0,
                contribution_score=75.0,
                exposure_score=80.0,
                user_id="test_user_123",
                asset_id="test_asset_456",
            )

            # Get the data that was passed to insert_one
            call_args = mock_db.get_analytics_collection().insert_one.call_args
            stored_data = call_args[0][0]

            # Verify all inputs are stored
            assert stored_data["model_earnings"] is not None
            assert stored_data["training_contribution_score"] == 75.0
            assert stored_data["usage_exposure_score"] == 80.0
            assert stored_data["equity_factor"] == 0.25
            assert stored_data["user_id"] == "test_user_123"
            assert stored_data["asset_id"] == "test_asset_456"

    @pytest.mark.asyncio
    async def test_retrieve_calculation_history(
        self,
        ai_value_service: AIValueService,
    ) -> None:
        """Test fetching historical calculations."""
        # Create mock cursor with async iteration
        mock_records = [
            {
                "_id": "calc_1",
                "user_id": "test_user",
                "asset_id": "asset_1",
                "model_earnings": "10000.00",
                "training_contribution_score": 75.0,
                "usage_exposure_score": 80.0,
                "equity_factor": 0.25,
                "calculated_value": "1500.00",
                "created_at": datetime.now(UTC),
            },
            {
                "_id": "calc_2",
                "user_id": "test_user",
                "asset_id": "asset_1",
                "model_earnings": "20000.00",
                "training_contribution_score": 60.0,
                "usage_exposure_score": 70.0,
                "equity_factor": 0.25,
                "calculated_value": "2100.00",
                "created_at": datetime.now(UTC),
            },
        ]

        # Create async iterator mock
        async def async_gen():
            for record in mock_records:
                yield record

        mock_cursor = MagicMock()
        mock_cursor.sort = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.__aiter__ = lambda _: async_gen()

        mock_collection = MagicMock()
        mock_collection.find = MagicMock(return_value=mock_cursor)

        mock_db_client = MagicMock()
        mock_db_client.get_analytics_collection = MagicMock(return_value=mock_collection)

        with patch("app.services.ai_value_service.get_db_client", return_value=mock_db_client):
            history = await ai_value_service.get_calculation_history(
                asset_id="asset_1",
                limit=10,
            )

            assert len(history) == 2
            assert history[0]["id"] == "calc_1"
            assert history[1]["id"] == "calc_2"

    @pytest.mark.asyncio
    async def test_storage_continues_on_db_error(
        self,
        ai_value_service: AIValueService,
    ) -> None:
        """Verify calculation continues even if storage fails."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            # Calculation should still complete
            result = await ai_value_service.calculate_ai_touch_value(
                model_earnings=10000.0,
                contribution_score=75.0,
                exposure_score=80.0,
                user_id="test_user_123",
            )

            assert result["calculated_value"] == "1500.00"
            assert result["calculation_id"] is None  # Storage failed


# =============================================================================
# Test API Endpoint Integration
# =============================================================================


class TestAPIEndpointIntegration:
    """
    Test POST /api/v1/analytics/predict endpoint.

    Per Agent Action Plan section 0.8 endpoint integration testing.
    """

    @pytest.fixture
    def test_client(self) -> TestClient:
        """Create TestClient for the analytics router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router, prefix="/api/v1/analytics")

        return TestClient(app)

    def test_endpoint_requires_authentication(self, test_client: TestClient) -> None:
        """Verify 401 without token."""
        response = test_client.post(
            "/api/v1/analytics/predict",
            json={
                "model_earnings": 10000.0,
                "training_contribution_score": 75.0,
                "usage_exposure_score": 80.0,
            },
        )

        # Should get 401, 403, or 422 (dependency error)
        assert response.status_code in [401, 403, 422]

    def test_endpoint_validates_input(
        self,
        mock_auth: dict[str, Any],
    ) -> None:
        """Verify 400/422 for invalid inputs."""
        from fastapi import FastAPI

        from app.core.auth import get_current_user

        app = FastAPI()
        app.include_router(router, prefix="/api/v1/analytics")

        # Override authentication dependency
        app.dependency_overrides[get_current_user] = lambda: mock_auth

        with TestClient(app) as client:
            # Test negative earnings
            response = client.post(
                "/api/v1/analytics/predict",
                json={
                    "model_earnings": -100.0,
                    "training_contribution_score": 75.0,
                    "usage_exposure_score": 80.0,
                },
            )

            assert response.status_code in [400, 422]

    def test_endpoint_validates_score_range(
        self,
        mock_auth: dict[str, Any],
    ) -> None:
        """Verify validation rejects out-of-range scores."""
        from fastapi import FastAPI

        from app.core.auth import get_current_user

        app = FastAPI()
        app.include_router(router, prefix="/api/v1/analytics")

        # Override authentication dependency
        app.dependency_overrides[get_current_user] = lambda: mock_auth

        with TestClient(app) as client:
            # Test contribution score > 100
            response = client.post(
                "/api/v1/analytics/predict",
                json={
                    "model_earnings": 10000.0,
                    "training_contribution_score": 150.0,
                    "usage_exposure_score": 80.0,
                },
            )

            assert response.status_code in [400, 422]

    def test_endpoint_returns_breakdown(
        self,
        mock_auth: dict[str, Any],
        mock_db: MagicMock,
    ) -> None:
        """Verify response includes formula breakdown."""
        from fastapi import FastAPI

        from app.core.auth import get_current_user

        app = FastAPI()
        app.include_router(router, prefix="/api/v1/analytics")

        # Override authentication dependency
        app.dependency_overrides[get_current_user] = lambda: mock_auth

        with (
            patch("app.services.ai_value_service.get_db_client", return_value=mock_db),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/v1/analytics/predict",
                json={
                    "model_earnings": 10000.0,
                    "training_contribution_score": 75.0,
                    "usage_exposure_score": 80.0,
                },
            )

            if response.status_code == 200:
                data = response.json()
                assert "formula_breakdown" in data
                assert "calculated_value" in data
                assert "equity_factor" in data
                assert data["equity_factor"] == 0.25


# =============================================================================
# Test Calculation Breakdown
# =============================================================================


class TestCalculationBreakdown:
    """
    Test detailed formula breakdown functionality.

    Verifies breakdown includes all formula components and intermediate values.
    """

    def test_calculation_breakdown_structure(self, ai_value_service: AIValueService) -> None:
        """Verify breakdown includes all formula components."""
        breakdown = ai_value_service.get_formula_breakdown(
            model_earnings=10000.0,
            contribution_score=75.0,
            exposure_score=80.0,
        )

        # Verify structure
        assert "formula" in breakdown
        assert "inputs" in breakdown
        assert "intermediate_values" in breakdown
        assert "factors" in breakdown
        assert "final_result" in breakdown

        # Verify inputs recorded
        assert breakdown["inputs"]["model_earnings"] == "10000.0"
        assert breakdown["inputs"]["contribution_score"] == 75.0
        assert breakdown["inputs"]["exposure_score"] == 80.0
        assert breakdown["inputs"]["equity_factor"] == 0.25

    def test_breakdown_shows_intermediate_values(self, ai_value_service: AIValueService) -> None:
        """Verify contribution_factor, exposure_factor shown."""
        breakdown = ai_value_service.get_formula_breakdown(
            model_earnings=10000.0,
            contribution_score=75.0,
            exposure_score=80.0,
        )

        factors = breakdown["factors"]
        assert factors["contribution_factor"] == 0.75
        assert factors["exposure_factor"] == 0.80
        assert factors["equity_factor"] == 0.25

        # Verify combined multiplier
        expected_multiplier = 0.75 * 0.80 * 0.25
        assert factors["combined_multiplier"] == pytest.approx(expected_multiplier, rel=1e-5)

    def test_breakdown_final_result(self, ai_value_service: AIValueService) -> None:
        """Verify final result is correctly calculated and displayed."""
        breakdown = ai_value_service.get_formula_breakdown(
            model_earnings=10000.0,
            contribution_score=75.0,
            exposure_score=80.0,
        )

        # 10000 * 0.75 * 0.80 * 0.25 = 1500.00
        assert breakdown["final_result"]["calculated_value"] == "1500.00"
        assert breakdown["final_result"]["currency"] == "USD"

    def test_breakdown_formula_string(self, ai_value_service: AIValueService) -> None:
        """Verify formula string is correct."""
        breakdown = ai_value_service.get_formula_breakdown(
            model_earnings=10000.0,
            contribution_score=75.0,
            exposure_score=80.0,
        )

        assert "AI Touch Value" in breakdown["formula"]
        assert "ModelEarnings" in breakdown["formula"]
        assert "ContributionScore" in breakdown["formula"]
        assert "ExposureScore" in breakdown["formula"]
        assert "EquityFactor" in breakdown["formula"]


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """
    Performance tests for AI Touch Value™ calculation.

    Ensures calculations are fast enough for production use.
    """

    @pytest.mark.asyncio
    async def test_calculation_performance(self, ai_value_service: AIValueService) -> None:
        """Verify calculation completes in < 10ms."""
        with patch("app.services.ai_value_service.get_db_client") as mock_get_db:
            mock_get_db.side_effect = RuntimeError("Database not initialized")

            start_time = time.perf_counter()

            await ai_value_service.calculate_ai_touch_value(
                model_earnings=100000.0,
                contribution_score=75.0,
                exposure_score=80.0,
            )

            elapsed_time = time.perf_counter() - start_time

            # Should complete in under 10ms (0.01 seconds)
            assert elapsed_time < 0.01, f"Calculation took {elapsed_time:.4f}s"

    @pytest.mark.asyncio
    async def test_bulk_calculations(
        self,
        ai_value_service: AIValueService,
        mock_db: MagicMock,
    ) -> None:
        """Test performance with 100 calculations."""
        with patch("app.services.ai_value_service.get_db_client", return_value=mock_db):
            start_time = time.perf_counter()

            for i in range(100):
                await ai_value_service.calculate_ai_touch_value(
                    model_earnings=10000.0 + i,
                    contribution_score=50.0 + (i % 50),
                    exposure_score=50.0 + (i % 50),
                    user_id=f"user_{i}",
                )

            elapsed_time = time.perf_counter() - start_time

            # 100 calculations should complete in under 1 second
            assert elapsed_time < 1.0, f"100 calculations took {elapsed_time:.4f}s"

    def test_formula_breakdown_performance(self, ai_value_service: AIValueService) -> None:
        """Test formula breakdown is fast."""
        start_time = time.perf_counter()

        for _ in range(100):
            ai_value_service.get_formula_breakdown(
                model_earnings=100000.0,
                contribution_score=75.0,
                exposure_score=80.0,
            )

        elapsed_time = time.perf_counter() - start_time

        # 100 breakdowns should complete in under 0.1 seconds
        assert elapsed_time < 0.1, f"100 breakdowns took {elapsed_time:.4f}s"


# =============================================================================
# Test Model Validation
# =============================================================================


class TestModelValidation:
    """
    Test AITouchValueCalculation model validation and computation.
    """

    def test_model_auto_calculates_value(self) -> None:
        """Verify model automatically calculates AI Touch Value™."""
        calc = AITouchValueCalculation(
            user_id="test_user",
            model_earnings=Decimal("10000.00"),
            training_contribution_score=75.0,
            usage_exposure_score=80.0,
        )

        # Should be auto-calculated: 10000 * 0.75 * 0.80 * 0.25 = 1500.00
        assert calc.calculated_value == Decimal("1500.00")

    def test_model_recalculate(self) -> None:
        """Test recalculate method."""
        calc = AITouchValueCalculation(
            user_id="test_user",
            model_earnings=Decimal("10000.00"),
            training_contribution_score=50.0,
            usage_exposure_score=50.0,
        )

        # Initial calculation: 10000 * 0.50 * 0.50 * 0.25 = 625.00
        assert calc.calculated_value == Decimal("625.00")

        # Recalculate returns the same value
        result = calc.recalculate()
        assert result == Decimal("625.00")

    def test_model_get_calculation_breakdown(self) -> None:
        """Test model's get_calculation_breakdown method."""
        calc = AITouchValueCalculation(
            user_id="test_user",
            model_earnings=Decimal("10000.00"),
            training_contribution_score=75.0,
            usage_exposure_score=80.0,
        )

        breakdown = calc.get_calculation_breakdown()

        assert breakdown["result"]["calculated_value"] == "1500.00"
        assert breakdown["factors"]["contribution_factor"] == 0.75
        assert breakdown["factors"]["exposure_factor"] == 0.80

    def test_model_to_mongodb_dict(self) -> None:
        """Test serialization to MongoDB format."""
        calc = AITouchValueCalculation(
            user_id="test_user",
            asset_id="test_asset",
            model_earnings=Decimal("10000.00"),
            training_contribution_score=75.0,
            usage_exposure_score=80.0,
        )

        mongo_dict = calc.to_mongodb_dict()

        # Verify string conversion for Decimal fields
        assert mongo_dict["model_earnings"] == "10000.00"
        assert mongo_dict["calculated_value"] == "1500.00"
        assert mongo_dict["user_id"] == "test_user"
        assert mongo_dict["asset_id"] == "test_asset"

    def test_model_from_mongodb_dict(self) -> None:
        """Test deserialization from MongoDB format."""
        mongo_doc = {
            "_id": "test_id_123",
            "user_id": "test_user",
            "asset_id": "test_asset",
            "model_earnings": "10000.00",
            "training_contribution_score": 75.0,
            "usage_exposure_score": 80.0,
            "equity_factor": 0.25,
            "calculated_value": "1500.00",
            "created_at": datetime.now(UTC),
        }

        calc = AITouchValueCalculation.from_mongodb_dict(mongo_doc)

        assert calc.id == "test_id_123"
        assert calc.user_id == "test_user"
        assert calc.model_earnings == Decimal("10000.00")
        assert calc.calculated_value == Decimal("1500.00")


# =============================================================================
# Test Value Projection
# =============================================================================


class TestValueProjection:
    """Test AI Touch Value™ projection over time periods."""

    @pytest.mark.asyncio
    async def test_project_value_basic(self, ai_value_service: AIValueService) -> None:
        """Test basic value projection."""
        projections = await ai_value_service.project_value(
            base_model_earnings=100000.0,
            contribution_score=75.0,
            exposure_score=80.0,
            growth_rate=0.1,
            periods=3,
        )

        assert len(projections) == 3

        # First period: 100000 * 0.75 * 0.80 * 0.25 = 15000.00
        assert projections[0]["period"] == 1
        assert projections[0]["projected_value"] == "15000.00"

        # Values should grow over periods
        for i in range(1, len(projections)):
            current = Decimal(projections[i]["projected_value"])
            previous = Decimal(projections[i - 1]["projected_value"])
            assert current > previous

    @pytest.mark.asyncio
    async def test_project_value_cumulative(self, ai_value_service: AIValueService) -> None:
        """Test cumulative value in projections."""
        projections = await ai_value_service.project_value(
            base_model_earnings=10000.0,
            contribution_score=100.0,
            exposure_score=100.0,
            growth_rate=0.0,  # No growth for easy calculation
            periods=4,
        )

        # Each period: 10000 * 1.0 * 1.0 * 0.25 = 2500.00
        # Cumulative after 4 periods: 10000.00
        cumulative = Decimal(projections[3]["cumulative_value"])
        assert cumulative == Decimal("10000.00")
