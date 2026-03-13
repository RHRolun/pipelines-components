"""Tests for the timeseries_forecasting component."""

import sys
from pathlib import Path
from unittest import mock

import pytest

# Inject mock modules so @mock.patch("pandas...") and @mock.patch("autogluon...") resolve
if "pandas" not in sys.modules:
    sys.modules["pandas"] = mock.MagicMock()
if "autogluon" not in sys.modules:
    _ag = mock.MagicMock()
    _ag.__path__ = []
    _ag.__spec__ = None
    sys.modules["autogluon"] = _ag
    _m = mock.MagicMock()
    _m.__spec__ = None
    sys.modules["autogluon.timeseries"] = _m

from ..component import timeseries_forecasting  # noqa: E402


def _make_mock_timeseries_dataframe():
    """Create a mock TimeSeriesDataFrame."""
    mock_df = mock.MagicMock()
    mock_df.columns = ["sales", "is_holiday", "promotion"]
    mock_df.num_items = 5
    mock_df.__len__.return_value = 1000
    return mock_df


def _make_mock_leaderboard(model_names):
    """Mock leaderboard DataFrame."""
    mock_lb = mock.MagicMock()
    mock_row = mock.MagicMock()
    mock_row.__getitem__.return_value = model_names[0]
    mock_lb.iloc = [mock_row]
    return mock_lb


def _make_mock_predictions():
    """Create mock predictions DataFrame."""
    mock_pred = mock.MagicMock()
    mock_pred.shape = (100, 10)  # 100 timesteps, 10 columns (mean + quantiles)
    return mock_pred


class TestTimeSeriesForecastingUnitTests:
    """Unit tests for component logic."""

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_timeseries_forecasting_with_basic_config(self, mock_ts_df_class, mock_predictor_class, mock_read_parquet):
        """Test basic time series forecasting with default parameters."""
        # Setup mocks
        mock_data = _make_mock_timeseries_dataframe()
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "MASE"
        mock_predictor.predict.return_value = _make_mock_predictions()
        mock_predictor.evaluate.return_value = {"MASE": 0.85, "MAPE": 0.12}
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["WeightedEnsemble"])
        mock_predictor.clone.return_value = mock_predictor
        mock_predictor_class.return_value = mock_predictor

        # Create mock artifacts
        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        workspace_path = "/tmp/workspace"

        # Call the component function
        result = timeseries_forecasting.python_func(
            timeseries_data=mock_timeseries_data,
            prediction_length=24,
            target_column="sales",
            workspace_path=workspace_path,
            model_artifact=mock_model_artifact,
            predictions=mock_predictions,
            metrics=mock_metrics,
        )

        # Verify read_parquet was called
        mock_read_parquet.assert_called_once_with("/tmp/timeseries_data.parquet")

        # Verify TimeSeriesPredictor was created with correct parameters
        mock_predictor_class.assert_called_once()
        call_kwargs = mock_predictor_class.call_args[1]
        assert call_kwargs["prediction_length"] == 24
        assert call_kwargs["target"] == "sales"
        assert call_kwargs["eval_metric"] == "MASE"
        assert call_kwargs["path"] == str(Path(workspace_path) / "timeseries_predictor")

        # Verify fit was called
        mock_predictor.fit.assert_called_once()

        # Verify predict was called
        mock_predictor.predict.assert_called_once()

        # Verify evaluate was called
        mock_predictor.evaluate.assert_called_once()

        # Verify return values
        assert result.predictor_path == str(Path(workspace_path) / "timeseries_predictor")
        assert result.best_model_name == "WeightedEnsemble"
        assert result.eval_metric_value == 0.85

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_timeseries_forecasting_with_known_covariates(
        self, mock_ts_df_class, mock_predictor_class, mock_read_parquet
    ):
        """Test forecasting with known covariates."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        # Mock __getitem__ for column selection
        mock_data.__getitem__.return_value = mock.MagicMock()

        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "MASE"
        mock_predictor.predict.return_value = _make_mock_predictions()
        mock_predictor.evaluate.return_value = {"MASE": 0.75}
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["DeepAR"])
        mock_predictor.clone.return_value = mock_predictor
        mock_predictor_class.return_value = mock_predictor

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Call with known covariates
        result = timeseries_forecasting.python_func(
            timeseries_data=mock_timeseries_data,
            prediction_length=24,
            target_column="sales",
            workspace_path="/tmp/workspace",
            model_artifact=mock_model_artifact,
            predictions=mock_predictions,
            metrics=mock_metrics,
            known_covariates_names=["is_holiday", "promotion"],
        )

        # Verify fit was called with known_covariates_names
        mock_predictor.fit.assert_called_once()
        fit_kwargs = mock_predictor.fit.call_args[1]
        assert fit_kwargs["known_covariates_names"] == ["is_holiday", "promotion"]

        # Verify predict was called with known_covariates
        assert mock_predictor.predict.called

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_timeseries_forecasting_with_different_presets(
        self, mock_ts_df_class, mock_predictor_class, mock_read_parquet
    ):
        """Test forecasting with different quality presets."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "MASE"
        mock_predictor.predict.return_value = _make_mock_predictions()
        mock_predictor.evaluate.return_value = {"MASE": 0.80}
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["ETS"])
        mock_predictor.clone.return_value = mock_predictor
        mock_predictor_class.return_value = mock_predictor

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Test with high_quality preset
        result = timeseries_forecasting.python_func(
            timeseries_data=mock_timeseries_data,
            prediction_length=24,
            target_column="sales",
            workspace_path="/tmp/workspace",
            model_artifact=mock_model_artifact,
            predictions=mock_predictions,
            metrics=mock_metrics,
            preset="high_quality",
            time_limit=7200,
        )

        # Verify fit was called with correct preset and time_limit
        mock_predictor.fit.assert_called_once()
        fit_kwargs = mock_predictor.fit.call_args[1]
        assert fit_kwargs["presets"] == "high_quality"
        assert fit_kwargs["time_limit"] == 7200

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_timeseries_forecasting_with_excluded_models(
        self, mock_ts_df_class, mock_predictor_class, mock_read_parquet
    ):
        """Test forecasting with excluded model types."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "MASE"
        mock_predictor.predict.return_value = _make_mock_predictions()
        mock_predictor.evaluate.return_value = {"MASE": 0.90}
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["ARIMA"])
        mock_predictor.clone.return_value = mock_predictor
        mock_predictor_class.return_value = mock_predictor

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Test with excluded models
        result = timeseries_forecasting.python_func(
            timeseries_data=mock_timeseries_data,
            prediction_length=24,
            target_column="sales",
            workspace_path="/tmp/workspace",
            model_artifact=mock_model_artifact,
            predictions=mock_predictions,
            metrics=mock_metrics,
            excluded_model_types=["DeepAR", "Transformer"],
        )

        # Verify fit was called with excluded_model_types
        mock_predictor.fit.assert_called_once()
        fit_kwargs = mock_predictor.fit.call_args[1]
        assert fit_kwargs["excluded_model_types"] == ["DeepAR", "Transformer"]

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_quantile_levels_configuration(self, mock_ts_df_class, mock_predictor_class, mock_read_parquet):
        """Test custom quantile levels configuration."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "MASE"
        mock_predictor.predict.return_value = _make_mock_predictions()
        mock_predictor.evaluate.return_value = {"MASE": 0.85}
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["WeightedEnsemble"])
        mock_predictor.clone.return_value = mock_predictor
        mock_predictor_class.return_value = mock_predictor

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Test with custom quantile levels
        custom_quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        result = timeseries_forecasting.python_func(
            timeseries_data=mock_timeseries_data,
            prediction_length=24,
            target_column="sales",
            workspace_path="/tmp/workspace",
            model_artifact=mock_model_artifact,
            predictions=mock_predictions,
            metrics=mock_metrics,
            quantile_levels=custom_quantiles,
        )

        # Verify TimeSeriesPredictor was created with custom quantiles
        call_kwargs = mock_predictor_class.call_args[1]
        assert call_kwargs["quantile_levels"] == custom_quantiles

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_artifact_metadata_saved_correctly(self, mock_ts_df_class, mock_predictor_class, mock_read_parquet):
        """Test that artifact metadata is saved with correct structure."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_predictor = mock.MagicMock()
        mock_predictor.eval_metric = "MASE"
        mock_predictor.predict.return_value = _make_mock_predictions()
        mock_predictor.evaluate.return_value = {"MASE": 0.85, "MAPE": 0.12}
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["WeightedEnsemble"])
        mock_predictor.clone.return_value = mock_predictor
        mock_predictor_class.return_value = mock_predictor

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Call the component
        result = timeseries_forecasting.python_func(
            timeseries_data=mock_timeseries_data,
            prediction_length=24,
            target_column="sales",
            workspace_path="/tmp/workspace",
            model_artifact=mock_model_artifact,
            predictions=mock_predictions,
            metrics=mock_metrics,
        )

        # Verify metadata structure
        assert "display_name" in mock_model_artifact.metadata
        assert "context" in mock_model_artifact.metadata
        assert "model_config" in mock_model_artifact.metadata["context"]
        assert "prediction_length" in mock_model_artifact.metadata["context"]
        assert mock_model_artifact.metadata["context"]["prediction_length"] == 24
        assert "target_column" in mock_model_artifact.metadata["context"]
        assert mock_model_artifact.metadata["context"]["target_column"] == "sales"
        assert "metrics" in mock_model_artifact.metadata["context"]
        assert "location" in mock_model_artifact.metadata["context"]

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_error_handling_missing_target_column(self, mock_ts_df_class, mock_predictor_class, mock_read_parquet):
        """Test error handling when target column is missing."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_data.columns = ["other_column"]  # Missing 'sales' column
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Verify ValueError is raised
        with pytest.raises(ValueError, match="Target column 'sales' not found"):
            timeseries_forecasting.python_func(
                timeseries_data=mock_timeseries_data,
                prediction_length=24,
                target_column="sales",
                workspace_path="/tmp/workspace",
                model_artifact=mock_model_artifact,
                predictions=mock_predictions,
                metrics=mock_metrics,
            )

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_error_handling_training_failure(self, mock_ts_df_class, mock_predictor_class, mock_read_parquet):
        """Test error handling when training fails."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_predictor = mock.MagicMock()
        mock_predictor.fit.side_effect = Exception("Training failed due to insufficient data")
        mock_predictor_class.return_value = mock_predictor

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Verify ValueError is raised
        with pytest.raises(ValueError, match="TimeSeriesPredictor training failed"):
            timeseries_forecasting.python_func(
                timeseries_data=mock_timeseries_data,
                prediction_length=24,
                target_column="sales",
                workspace_path="/tmp/workspace",
                model_artifact=mock_model_artifact,
                predictions=mock_predictions,
                metrics=mock_metrics,
            )

    @mock.patch("pandas.read_parquet")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    def test_error_handling_missing_known_covariates(self, mock_ts_df_class, mock_predictor_class, mock_read_parquet):
        """Test error handling when specified known covariates don't exist."""
        mock_data = _make_mock_timeseries_dataframe()
        mock_data.columns = ["sales"]  # Missing 'is_holiday', 'promotion'
        mock_read_parquet.return_value = mock_data
        mock_ts_df_class.return_value = mock_data

        mock_timeseries_data = mock.MagicMock()
        mock_timeseries_data.path = "/tmp/timeseries_data.parquet"

        mock_model_artifact = mock.MagicMock()
        mock_model_artifact.path = "/tmp/model"
        mock_model_artifact.metadata = {}

        mock_predictions = mock.MagicMock()
        mock_predictions.path = "/tmp/predictions.parquet"

        mock_metrics = mock.MagicMock()
        mock_metrics.path = "/tmp/metrics.json"

        # Verify ValueError is raised
        with pytest.raises(ValueError, match="Known covariates .* not found"):
            timeseries_forecasting.python_func(
                timeseries_data=mock_timeseries_data,
                prediction_length=24,
                target_column="sales",
                workspace_path="/tmp/workspace",
                model_artifact=mock_model_artifact,
                predictions=mock_predictions,
                metrics=mock_metrics,
                known_covariates_names=["is_holiday", "promotion"],
            )

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(timeseries_forecasting)
        assert hasattr(timeseries_forecasting, "python_func")
        assert hasattr(timeseries_forecasting, "component_spec")
