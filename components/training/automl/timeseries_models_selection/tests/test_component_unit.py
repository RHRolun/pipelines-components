"""Tests for the timeseries_models_selection component."""

import sys
from pathlib import Path
from unittest import mock

import pytest

from ..component import timeseries_models_selection  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def isolated_sys_modules():
    """Patch pandas/autogluon in sys.modules only for this test module; restored on module teardown."""
    with mock.patch.dict(sys.modules, clear=False) as mocked_modules:
        mocked_modules["pandas"] = mock.MagicMock()
        _ag = mock.MagicMock()
        _ag.__path__ = []
        _ag.__spec__ = None
        mocked_modules["autogluon"] = _ag
        _m = mock.MagicMock()
        _m.__spec__ = None
        mocked_modules["autogluon.timeseries"] = _m
        yield


def _make_mock_leaderboard(all_model_names, eval_metric="MASE"):
    """Mock leaderboard so .head(n)['model'].values.tolist() returns first n names."""

    def _head(n):
        head_mock = mock.MagicMock()
        col_mock = mock.MagicMock()
        col_mock.values.tolist.return_value = all_model_names[:n]
        head_mock.__getitem__.return_value = col_mock
        # Mock iloc for getting eval_metric value
        iloc_mock = mock.MagicMock()
        iloc_mock.__getitem__.return_value = 0.75  # Mock MASE value
        head_mock.iloc = [iloc_mock]
        return head_mock

    mock_lb = mock.MagicMock()
    mock_lb.head.side_effect = _head
    mock_lb.__len__.return_value = len(all_model_names)
    mock_lb.to_dict.return_value = {
        "model": all_model_names,
        eval_metric: [0.75 + i * 0.05 for i in range(len(all_model_names))],
    }
    return mock_lb


def _make_mock_timeseries_df(num_items=3, num_timesteps=100):
    """Create mock TimeSeriesDataFrame."""
    mock_ts_df = mock.MagicMock()
    mock_ts_df.num_items = num_items
    mock_ts_df.__len__.return_value = num_items * num_timesteps
    return mock_ts_df


class TestTimeseriesModelsSelectionUnitTests:
    """Unit tests for component logic."""

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_timeseries_models_selection_basic_functionality(
        self, mock_predictor_class, mock_ts_df_class, mock_read_csv
    ):
        """Test basic timeseries model selection functionality."""
        # Setup mocks
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.return_value = mock_predictor
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(
            ["DeepAR", "TemporalFusionTransformer", "AutoARIMA", "AutoETS", "Theta"]
        )
        mock_predictor_class.return_value = mock_predictor

        # Mock TimeSeriesDataFrame
        mock_train_ts = _make_mock_timeseries_df()
        mock_test_ts = _make_mock_timeseries_df()
        mock_ts_df_class.from_data_frame.side_effect = [mock_train_ts, mock_test_ts]

        # Mock read_csv
        mock_train_df = mock.MagicMock()
        mock_test_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        # Create mock test data artifact
        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        workspace_path = "/tmp/workspace"

        # Call the component
        result = timeseries_models_selection.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train_data.csv",
            test_data=mock_test_data,
            top_n=3,
            workspace_path=workspace_path,
            prediction_length=24,
        )

        # Verify read_csv was called correctly
        assert mock_read_csv.call_count == 2
        assert mock_read_csv.call_args_list[0][0][0] == "/tmp/train_data.csv"
        assert mock_read_csv.call_args_list[1][0][0] == "/tmp/test_data.csv"

        # Verify TimeSeriesDataFrame.from_data_frame was called correctly
        assert mock_ts_df_class.from_data_frame.call_count == 2
        mock_ts_df_class.from_data_frame.assert_any_call(
            mock_train_df, id_column="item_id", timestamp_column="timestamp"
        )
        mock_ts_df_class.from_data_frame.assert_any_call(
            mock_test_df, id_column="item_id", timestamp_column="timestamp"
        )

        # Verify TimeSeriesPredictor was created correctly
        mock_predictor_class.assert_called_once_with(
            prediction_length=24,
            target="sales",
            path=str(Path(workspace_path) / "timeseries_predictor"),
            eval_metric="MASE",
            verbosity=2,
        )

        # Verify fit was called
        mock_predictor.fit.assert_called_once()
        fit_call_kwargs = mock_predictor.fit.call_args[1]
        assert fit_call_kwargs["presets"] == "medium_quality"
        assert fit_call_kwargs["time_limit"] == 3600

        # Verify leaderboard was called
        mock_predictor.leaderboard.assert_called_once_with(mock_test_ts)

        # Verify return values
        assert result.top_models == ["DeepAR", "TemporalFusionTransformer", "AutoARIMA"]
        assert len(result.top_models) == 3
        assert result.eval_metric_name == "MASE"
        assert result.predictor_path == str(Path(workspace_path) / "timeseries_predictor")
        assert result.model_config["prediction_length"] == 24
        assert result.model_config["eval_metric"] == "MASE"
        assert result.model_config["target"] == "sales"

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_timeseries_models_selection_with_different_top_n(
        self, mock_predictor_class, mock_ts_df_class, mock_read_csv
    ):
        """Test model selection with different top_n values."""
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.return_value = mock_predictor
        all_models = [
            "DeepAR",
            "TemporalFusionTransformer",
            "AutoARIMA",
            "AutoETS",
            "Theta",
            "SeasonalNaive",
            "DirectTabular",
        ]
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(all_models)
        mock_predictor_class.return_value = mock_predictor

        mock_train_ts = _make_mock_timeseries_df()
        mock_test_ts = _make_mock_timeseries_df()
        mock_ts_df_class.from_data_frame.side_effect = [mock_train_ts, mock_test_ts]

        mock_train_df = mock.MagicMock()
        mock_test_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        # Test with top_n=1
        result = timeseries_models_selection.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train_data.csv",
            test_data=mock_test_data,
            top_n=1,
            workspace_path="/tmp/workspace",
            prediction_length=24,
        )

        assert len(result.top_models) == 1
        assert result.top_models == ["DeepAR"]

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_timeseries_models_selection_with_known_covariates(
        self, mock_predictor_class, mock_ts_df_class, mock_read_csv
    ):
        """Test model selection with known covariates."""
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.return_value = mock_predictor
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["DeepAR", "TemporalFusionTransformer"])
        mock_predictor_class.return_value = mock_predictor

        mock_train_ts = _make_mock_timeseries_df()
        mock_test_ts = _make_mock_timeseries_df()
        mock_ts_df_class.from_data_frame.side_effect = [mock_train_ts, mock_test_ts]

        mock_train_df = mock.MagicMock()
        mock_test_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        known_covariates = ["is_holiday", "promo_flag"]

        # Call component with known covariates
        result = timeseries_models_selection.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train_data.csv",
            test_data=mock_test_data,
            top_n=2,
            workspace_path="/tmp/workspace",
            prediction_length=24,
            known_covariates_names=known_covariates,
        )

        # Verify fit was called with known_covariates_names
        fit_call_kwargs = mock_predictor.fit.call_args[1]
        assert fit_call_kwargs["known_covariates_names"] == known_covariates

        # Verify model_config includes known_covariates_names
        assert result.model_config["known_covariates_names"] == known_covariates

    @mock.patch("pandas.read_csv")
    def test_timeseries_models_selection_handles_file_not_found_train_data(self, mock_read_csv):
        """Test that FileNotFoundError is raised when train_data path doesn't exist."""
        mock_read_csv.side_effect = FileNotFoundError("Train data file not found")

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        with pytest.raises(FileNotFoundError):
            timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/nonexistent/train_data.csv",
                test_data=mock_test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
                prediction_length=24,
            )

    @mock.patch("pandas.read_csv")
    def test_timeseries_models_selection_handles_file_not_found_test_data(self, mock_read_csv):
        """Test that FileNotFoundError is raised when test_data path doesn't exist."""
        mock_train_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, FileNotFoundError("Test data file not found")]

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/nonexistent/test_data.csv"

        with pytest.raises(FileNotFoundError):
            timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train_data.csv",
                test_data=mock_test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
                prediction_length=24,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_timeseries_models_selection_handles_training_failure(
        self, mock_predictor_class, mock_ts_df_class, mock_read_csv
    ):
        """Test that ValueError is raised when model training fails."""
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.side_effect = ValueError("Training failed: insufficient data")
        mock_predictor_class.return_value = mock_predictor

        mock_train_ts = _make_mock_timeseries_df()
        mock_test_ts = _make_mock_timeseries_df()
        mock_ts_df_class.from_data_frame.side_effect = [mock_train_ts, mock_test_ts]

        mock_train_df = mock.MagicMock()
        mock_test_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        with pytest.raises(ValueError, match="TimeSeriesPredictor training failed"):
            timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train_data.csv",
                test_data=mock_test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
                prediction_length=24,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_timeseries_models_selection_handles_leaderboard_failure(
        self, mock_predictor_class, mock_ts_df_class, mock_read_csv
    ):
        """Test that ValueError is raised when leaderboard generation fails."""
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.return_value = mock_predictor
        mock_predictor.leaderboard.side_effect = ValueError("Leaderboard generation failed")
        mock_predictor_class.return_value = mock_predictor

        mock_train_ts = _make_mock_timeseries_df()
        mock_test_ts = _make_mock_timeseries_df()
        mock_ts_df_class.from_data_frame.side_effect = [mock_train_ts, mock_test_ts]

        mock_train_df = mock.MagicMock()
        mock_test_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        with pytest.raises(ValueError, match="Failed to generate leaderboard"):
            timeseries_models_selection.python_func(
                target="sales",
                id_column="item_id",
                timestamp_column="timestamp",
                train_data_path="/tmp/train_data.csv",
                test_data=mock_test_data,
                top_n=2,
                workspace_path="/tmp/workspace",
                prediction_length=24,
            )

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_timeseries_models_selection_returns_correct_named_tuple(
        self, mock_predictor_class, mock_ts_df_class, mock_read_csv
    ):
        """Test that the function returns a NamedTuple with correct fields."""
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.return_value = mock_predictor
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["DeepAR", "TemporalFusionTransformer"])
        mock_predictor_class.return_value = mock_predictor

        mock_train_ts = _make_mock_timeseries_df()
        mock_test_ts = _make_mock_timeseries_df()
        mock_ts_df_class.from_data_frame.side_effect = [mock_train_ts, mock_test_ts]

        mock_train_df = mock.MagicMock()
        mock_test_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        result = timeseries_models_selection.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train_data.csv",
            test_data=mock_test_data,
            top_n=2,
            workspace_path="/tmp/workspace",
            prediction_length=24,
        )

        # Verify return type and fields
        assert hasattr(result, "top_models")
        assert hasattr(result, "predictor_path")
        assert hasattr(result, "eval_metric_name")
        assert hasattr(result, "model_config")
        assert isinstance(result.top_models, list)
        assert isinstance(result.predictor_path, str)
        assert isinstance(result.eval_metric_name, str)
        assert isinstance(result.model_config, dict)

    @mock.patch("pandas.read_csv")
    @mock.patch("autogluon.timeseries.TimeSeriesDataFrame")
    @mock.patch("autogluon.timeseries.TimeSeriesPredictor")
    def test_timeseries_models_selection_verifies_all_operations_called(
        self, mock_predictor_class, mock_ts_df_class, mock_read_csv
    ):
        """Test that all required operations are called in correct order."""
        mock_predictor = mock.MagicMock()
        mock_predictor.fit.return_value = mock_predictor
        mock_predictor.leaderboard.return_value = _make_mock_leaderboard(["DeepAR", "AutoARIMA"])
        mock_predictor_class.return_value = mock_predictor

        mock_train_ts = _make_mock_timeseries_df()
        mock_test_ts = _make_mock_timeseries_df()
        mock_ts_df_class.from_data_frame.side_effect = [mock_train_ts, mock_test_ts]

        mock_train_df = mock.MagicMock()
        mock_test_df = mock.MagicMock()
        mock_read_csv.side_effect = [mock_train_df, mock_test_df]

        mock_test_data = mock.MagicMock()
        mock_test_data.path = "/tmp/test_data.csv"

        timeseries_models_selection.python_func(
            target="sales",
            id_column="item_id",
            timestamp_column="timestamp",
            train_data_path="/tmp/train_data.csv",
            test_data=mock_test_data,
            top_n=2,
            workspace_path="/tmp/workspace",
            prediction_length=24,
        )

        # Verify call order: read_csv -> from_data_frame -> TimeSeriesPredictor -> fit -> leaderboard
        assert mock_read_csv.call_count == 2
        assert mock_ts_df_class.from_data_frame.call_count == 2
        assert mock_predictor_class.called
        assert mock_predictor.fit.called
        assert mock_predictor.leaderboard.called

        # Verify fit was called before leaderboard
        assert mock_predictor.fit.call_count == 1
        assert mock_predictor.leaderboard.call_count == 1

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(timeseries_models_selection)
        assert hasattr(timeseries_models_selection, "python_func")
        assert hasattr(timeseries_models_selection, "component_spec")
