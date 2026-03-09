# Autogluon Timeseries Training Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

AutoGluon Time Series Training Pipeline (draft design).

This pipeline is intended to train AutoGluon time series forecasting models on data loaded from S3, evaluate them on a
validation split, and output a leaderboard plus the top N trained predictors. Components are not yet implemented; this
docstring describes the target design and parameters.

**Intended pipeline stages**

1. **Data loading**: Load time series data from S3 (CSV/Parquet). The data must contain columns for item_id, timestamp,
and target (and optionally known covariates). Build a :class:`~autogluon.timeseries.TimeSeriesDataFrame` (multi-index on
item_id, timestamp).

2. **Train / validation split**: Chronological or expanding-window split so that validation is used for model selection
and evaluation (e.g. last ``prediction_length`` steps or configurable holdout). Produces train and validation
TimeSeriesDataFrames.

3. **Training and model selection**: Train multiple AutoGluon TimeSeries models (local e.g. ARIMA, ETS, Theta; global
e.g. DeepAR, TFT) on the training data. Rank by eval_metric (e.g. WQL, MASE) on the validation set and select the top N
models.

4. **Leaderboard evaluation**: Aggregate metrics from the trained models and produce an HTML leaderboard ranking them by
the chosen evaluation metric. Output the top N predictors (and optionally metrics, notebook) for deployment.

**Parameters (data and identity)**

- **train_data_secret_name**: Kubernetes secret name containing S3 credentials (e.g. AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION). - **train_data_bucket_name**: S3-compatible bucket name
containing the time series data file. - **train_data_file_key**: S3 object key of the data file (CSV or Parquet). File
must include columns for item_id, timestamp, and target; optional columns for known covariates.

**Parameters (TimeSeriesDataFrame schema)**

- **target**: Name of the column containing the numeric values to forecast (the time series values). Corresponds to
:attr:`~autogluon.timeseries.TimeSeriesDataFrame` target column. - **id_column**: Name of the column that identifies
each time series (e.g. product_id, store_id). Passed as ``id_column`` when constructing TimeSeriesDataFrame from a
DataFrame or path; the result uses ``item_id`` as the first index level. - **timestamp_column**: Name of the column
containing the timestamp/datetime for each observation. Passed as ``timestamp_column`` when constructing
TimeSeriesDataFrame; the result uses ``timestamp`` as the second index level.

**Parameters (predictor and selection)**

- **known_covariates_names**: Optional list of column names that are known in advance for all steps in the forecast
horizon (e.g. holidays, promotions). If provided, the predictor expects these columns in the data and at prediction time
requires future values in ``known_covariates``. See
:attr:`~autogluon.timeseries.TimeSeriesPredictor.known_covariates_names`. - **freq**: Optional pandas frequency string
(e.g. ``"D"`` for daily, ``"h"`` for hourly). If not set, frequency is inferred from the data. Set when timestamps are
irregular or when resampling to a different frequency. See :attr:`~autogluon.timeseries.TimeSeriesPredictor.freq`. -
**top_n**: Number of top models to select for the leaderboard and output (default: 3). Positive integer.

**Returns**

Intended outputs (once implemented): leaderboard artifact (HTML), trained model artifacts (predictor and metrics for the
top N models), and optionally a generated notebook for inference.

**Raises**

Expected to raise on: missing or inaccessible S3 file; missing ``target``, ``id_column``, or ``timestamp_column`` in the
data; invalid ``freq`` or ``top_n``; or failure to build TimeSeriesDataFrame (e.g. duplicate item_id/timestamp pairs).

**Example**

pipeline = autogluon_timeseries_training_pipeline( train_data_secret_name="my-s3-secret",
train_data_bucket_name="my-bucket", train_data_file_key="ts/sales.csv", target="sales", id_column="product_id",
timestamp_column="date", known_covariates_names=["is_holiday", "promo"], freq="D", top_n=3, )

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_secret_name` | `str` | `None` |  |
| `train_data_bucket_name` | `str` | `None` |  |
| `train_data_file_key` | `str` | `None` |  |
| `target` | `str` | `None` |  |
| `id_column` | `str` | `None` |  |
| `timestamp_column` | `str` | `None` |  |
| `known_covariates_names` | `Optional[List[str]]` | `None` |  |
| `freq` | `Optional[str]` | `None` |  |
| `top_n` | `int` | `3` |  |

## Metadata 🗂️

- **Name**: autogluon_timeseries_training_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - pipeline
  - automl
  - autogluon-timeseries-training-pipeline
- **Last Verified**: 2026-03-09 17:05:29+00:00
- **Owners**:
  - Approvers:
    - DorotaDR
    - mateusz.switala
  - Reviewers:
    - DorotaDR
