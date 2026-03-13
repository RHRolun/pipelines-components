# AutoGluon Time Series Forecasting ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train AutoGluon TimeSeriesPredictor and generate probabilistic forecasts.

This component trains multiple time series forecasting models using AutoGluon's TimeSeriesPredictor, which automatically
selects and ensembles models including DeepAR, Transformer, ARIMA, ETS, Theta, and more. The component handles both
single and multiple time series, supports known covariates (future-available features), and generates probabilistic
forecasts with quantiles.

The TimeSeriesPredictor automatically trains various model types (deep learning, statistical, tree-based) and combines
them using a weighted ensemble. Unlike tabular data, time series forecasting requires temporal validation (last
prediction_length timesteps as test set) rather than random splits.

This component outputs a trained predictor, multi-step ahead forecasts with quantiles, and evaluation metrics. The
predictor can be integrated with other AutoML components like autogluon_leaderboard_evaluation for visualization.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeseries_data` | `dsl.Input[dsl.Dataset]` | `None` | TimeSeriesDataFrame in parquet format with multi-index (item_id, timestamp). |
| `prediction_length` | `int` | `None` | Forecast horizon (number of timesteps to predict ahead). Must be positive. |
| `target_column` | `str` | `None` | Name of the target variable to forecast. Must exist in timeseries_data. |
| `workspace_path` | `str` | `None` | Workspace directory where TimeSeriesPredictor is saved (workspace_path/timeseries_predictor). |
| `model_artifact` | `dsl.Output[dsl.Model]` | `None` | Output Model artifact containing trained predictor with metadata. |
| `predictions` | `dsl.Output[dsl.Dataset]` | `None` | Output Dataset artifact containing forecasts with quantiles (parquet format). |
| `metrics` | `dsl.Output[dsl.Metrics]` | `None` | Output Metrics artifact containing evaluation results (JSON format). |
| `eval_metric` | `str` | `"MASE"` | Evaluation metric for ranking models. Options: MASE, MAPE, SMAPE, MSE, RMSE, MAE, WQL. |
| `preset` | `str` | `"medium_quality"` | AutoGluon quality preset. Options: fast_training, medium_quality, high_quality, best_quality. |
| `time_limit` | `int` | `3600` | Training time limit in seconds (default: 3600 = 1 hour). |
| `known_covariates_names` | `Optional[List[str]]` | `None` | List of column names for known covariates (features known in advance). |
| `excluded_model_types` | `Optional[List[str]]` | `None` | List of model types to exclude from training (e.g., ["DeepAR", "Transformer"]). |
| `quantile_levels` | `Optional[List[float]]` | `None` | List of quantile levels for probabilistic forecasts. Default: [0.1, 0.2, ..., 0.9]. |
| `num_val_windows` | `int` | `1` | Number of validation windows for backtesting (default: 1). |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `NamedTuple('outputs', predictor_path=str, best_model_name=str, eval_metric_value=float)` | predictor_path (str), best_model_name (str), eval_metric_value (float). |

## Metadata 🗂️

- **Name**: timeseries_forecasting
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.14.4
- **Tags**:
  - training
  - automl
  - timeseries
  - forecasting
- **Last Verified**: 2026-03-06 00:00:00+00:00
- **Owners**:
  - Approvers:
    - mprahl
    - nsingla
  - Reviewers:
    - HumairAK

## Usage Examples 💡

### Basic usage (single or multiple time series)

Train time series forecasting models and generate predictions; use `dsl.WORKSPACE_PATH_PLACEHOLDER` for workspace in a pipeline:

```python
from kfp import dsl
from kfp_components.components.training.automl.timeseries_forecasting import timeseries_forecasting

@dsl.pipeline(name="timeseries-forecasting-pipeline")
def my_pipeline(timeseries_dataset):
    forecast_task = timeseries_forecasting(
        timeseries_data=timeseries_dataset,
        prediction_length=24,  # Forecast 24 hours ahead
        target_column="sales",
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        model_artifact=dsl.Output(type="Model"),
        predictions=dsl.Output(type="Dataset"),
        metrics=dsl.Output(type="Metrics"),
    )
    return forecast_task.outputs["predictor_path"], forecast_task.outputs["best_model_name"]
```

### With known covariates (holidays, promotions)

Include features that are known in advance to improve forecast accuracy:

```python
forecast_task = timeseries_forecasting(
    timeseries_data=timeseries_dataset,
    prediction_length=7,  # 7 days ahead
    target_column="demand",
    known_covariates_names=["is_holiday", "promotion_active", "day_of_week"],
    workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    model_artifact=dsl.Output(type="Model"),
    predictions=dsl.Output(type="Dataset"),
    metrics=dsl.Output(type="Metrics"),
    preset="high_quality",
    time_limit=7200,  # 2 hours
)
```

### Different quality presets

Trade off speed vs accuracy with different preset levels:

```python
# Fast training for quick experiments
fast_forecast = timeseries_forecasting(
    timeseries_data=timeseries_dataset,
    prediction_length=24,
    target_column="sales",
    preset="fast_training",  # Fastest, lower accuracy
    time_limit=600,  # 10 minutes
    workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    model_artifact=dsl.Output(type="Model"),
    predictions=dsl.Output(type="Dataset"),
    metrics=dsl.Output(type="Metrics"),
)

# Best quality for production
best_forecast = timeseries_forecasting(
    timeseries_data=timeseries_dataset,
    prediction_length=24,
    target_column="sales",
    preset="best_quality",  # Highest accuracy, slowest
    time_limit=14400,  # 4 hours
    workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    model_artifact=dsl.Output(type="Model"),
    predictions=dsl.Output(type="Dataset"),
    metrics=dsl.Output(type="Metrics"),
)
```

### Custom quantile levels for risk assessment

Specify quantile levels for probabilistic forecasts:

```python
forecast_task = timeseries_forecasting(
    timeseries_data=timeseries_dataset,
    prediction_length=30,  # 30 days ahead
    target_column="revenue",
    quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95],  # 5th, 25th, 50th, 75th, 95th percentiles
    workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    model_artifact=dsl.Output(type="Model"),
    predictions=dsl.Output(type="Dataset"),
    metrics=dsl.Output(type="Metrics"),
)
# Predictions will include columns: mean, 0.05, 0.25, 0.5, 0.75, 0.95
```

### Exclude specific model types

Exclude computationally expensive models for faster training:

```python
forecast_task = timeseries_forecasting(
    timeseries_data=timeseries_dataset,
    prediction_length=168,  # 1 week of hourly data
    target_column="energy_demand",
    excluded_model_types=["DeepAR", "Transformer"],  # Skip deep learning models
    workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    model_artifact=dsl.Output(type="Model"),
    predictions=dsl.Output(type="Dataset"),
    metrics=dsl.Output(type="Metrics"),
    preset="medium_quality",
)
```

## Key Differences from Tabular Forecasting 🔑

- **Temporal ordering matters**: Data must be sorted by timestamp; random shuffling breaks time dependencies
- **Multi-step ahead**: Predicts multiple future timesteps, not single values
- **Probabilistic forecasts**: Returns quantiles (e.g., 10th, 50th, 90th percentiles) for uncertainty estimation
- **Known covariates**: Future-available features (e.g., holidays, planned promotions) can improve forecasts
- **Temporal validation**: Uses last N timesteps for validation, not random split
- **Different metrics**: MASE, SMAPE, WQL instead of accuracy/RMSE

## Supported Time Series Models 🤖

AutoGluon TimeSeriesPredictor automatically trains and ensembles:

- **Deep Learning**: DeepAR, Transformer, SimpleFeedForward, TemporalFusionTransformer
- **Statistical**: ARIMA, ETS, Theta, AutoARIMA, AutoETS
- **Tree-based**: DirectTabular (converts time series to tabular format)
- **Naive baselines**: SeasonalNaive, Naive, Average
- **Ensemble**: WeightedEnsemble (combines all models based on validation performance)

## Data Format Requirements 📊

Input data should be a TimeSeriesDataFrame in parquet format with:

- **Multi-index**: `(item_id, timestamp)` where:
  - `item_id`: Unique identifier for each time series (e.g., store_id, product_id)
  - `timestamp`: Datetime index with consistent frequency
- **Columns**:
  - Target column (specified in `target_column` parameter)
  - Optional known covariates columns
  - Optional static features (constant per item_id)

Example structure:

```
                        sales  is_holiday  promotion
item_id  timestamp
store_1  2024-01-01      100           0          0
         2024-01-02      105           0          1
         2024-01-03      120           1          0
store_2  2024-01-01       50           0          0
         2024-01-02       55           0          1
         2024-01-03       60           1          0
```

## Integration with Other Components 🔗

This component integrates seamlessly with the existing AutoML ecosystem:

- **Leaderboard generation**: Use the existing `autogluon_leaderboard_evaluation` component to generate HTML leaderboards
  for time series models. The component expects model artifacts with metadata, which `timeseries_forecasting` provides.
- **Pipeline composition**: Chain with data processing components and evaluation components in a Kubeflow pipeline.

## References 📚

- [AutoGluon TimeSeriesPredictor Documentation](https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html)
- [Time Series Forecasting Tutorial](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html)
- [AutoGluon Time Series Quick Start](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-quick-start.html)
