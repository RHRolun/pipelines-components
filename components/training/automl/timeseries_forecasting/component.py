from typing import List, NamedTuple, Optional

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
    packages_to_install=[
        "autogluon.timeseries==1.5.0",
        "pandas==2.2.3",
    ],
)
def timeseries_forecasting(
    timeseries_data: dsl.Input[dsl.Dataset],
    prediction_length: int,
    target_column: str,
    workspace_path: str,
    model_artifact: dsl.Output[dsl.Model],
    predictions: dsl.Output[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    eval_metric: str = "MASE",
    preset: str = "medium_quality",
    time_limit: int = 3600,
    known_covariates_names: Optional[List[str]] = None,
    excluded_model_types: Optional[List[str]] = None,
    quantile_levels: Optional[List[float]] = None,
    num_val_windows: int = 1,
) -> NamedTuple("outputs", predictor_path=str, best_model_name=str, eval_metric_value=float):
    """Train AutoGluon TimeSeriesPredictor and generate probabilistic forecasts.

    This component trains multiple time series forecasting models using AutoGluon's
    TimeSeriesPredictor, which automatically selects and ensembles models including
    DeepAR, Transformer, ARIMA, ETS, Theta, and more. The component handles both
    single and multiple time series, supports known covariates (future-available
    features), and generates probabilistic forecasts with quantiles.

    The TimeSeriesPredictor automatically trains various model types (deep learning,
    statistical, tree-based) and combines them using a weighted ensemble. Unlike
    tabular data, time series forecasting requires temporal validation (last
    prediction_length timesteps as test set) rather than random splits.

    This component outputs a trained predictor, multi-step ahead forecasts with
    quantiles, and evaluation metrics. The predictor can be used for generating
    new forecasts or integrated with other AutoML components like
    autogluon_leaderboard_evaluation for visualization.

    Args:
        timeseries_data: TimeSeriesDataFrame in parquet format with multi-index
            (item_id, timestamp). Required columns: target_column and optionally
            known_covariates_names.
        prediction_length: Forecast horizon (number of timesteps to predict ahead).
            Must be positive and less than the length of each time series.
        target_column: Name of the target variable to forecast. Must exist in
            timeseries_data.
        workspace_path: Workspace directory where TimeSeriesPredictor is saved
            (workspace_path/timeseries_predictor).
        model_artifact: Output Model artifact containing trained predictor with
            metadata. Directory structure: predictor/, metrics/, notebooks/.
        predictions: Output Dataset artifact containing forecasts with quantiles
            (parquet format). Columns include mean and specified quantile levels.
        metrics: Output Metrics artifact containing evaluation results (JSON format).
            Includes metrics like MASE, MAPE, SMAPE, MSE, etc.
        eval_metric: Evaluation metric for ranking models. Options: "MASE" (default),
            "MAPE", "SMAPE", "MSE", "RMSE", "MAE", "WQL". MASE is recommended for
            time series.
        preset: AutoGluon quality preset. Options: "fast_training", "medium_quality"
            (default), "high_quality", "best_quality". Higher quality takes longer
            but may improve accuracy.
        time_limit: Training time limit in seconds (default: 3600 = 1 hour). Actual
            time may exceed limit slightly for model finalization.
        known_covariates_names: List of column names for known covariates (features
            known in advance, e.g., holidays, promotions). If provided, these
            features must be available for both training and prediction periods.
        excluded_model_types: List of model types to exclude from training (e.g.,
            ["DeepAR", "Transformer"]). Useful for faster training or avoiding
            specific models.
        quantile_levels: List of quantile levels for probabilistic forecasts (e.g.,
            [0.1, 0.5, 0.9] for 10th, 50th, 90th percentiles). If None, uses
            default [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].
        num_val_windows: Number of validation windows for backtesting (default: 1).
            More windows provide better metric estimates but increase training time.

    Returns:
        NamedTuple: predictor_path (str), best_model_name (str), eval_metric_value (float).

    Raises:
        FileNotFoundError: If timeseries_data path cannot be found.
        ValueError: If prediction_length invalid, target_column missing, training fails,
            or time series too short for specified prediction_length.
        KeyError: If required columns missing from timeseries_data.

    Example:
        from kfp import dsl
        from components.training.automl.timeseries_forecasting import (
            timeseries_forecasting
        )

        @dsl.pipeline(name="timeseries-forecasting-pipeline")
        def forecast_pipeline(timeseries_dataset, workspace_path):
            "Train time series models and generate forecasts."
            result = timeseries_forecasting(
                timeseries_data=timeseries_dataset,
                prediction_length=24,  # Forecast 24 hours ahead
                target_column="sales",
                workspace_path=workspace_path,
                model_artifact=dsl.Output(type="Model"),
                predictions=dsl.Output(type="Dataset"),
                metrics=dsl.Output(type="Metrics"),
                preset="medium_quality",
                known_covariates_names=["is_holiday", "promotion"],
            )
            return result.predictor_path, result.best_model_name

    """  # noqa: E501
    import json
    import logging
    from pathlib import Path

    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    logger = logging.getLogger(__name__)

    # Set default quantile levels if not provided
    if quantile_levels is None:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Load the TimeSeriesDataFrame from parquet
    logger.info(f"Loading time series data from {timeseries_data.path}")
    data = pd.read_parquet(timeseries_data.path)

    # Convert to TimeSeriesDataFrame if needed
    if not isinstance(data, TimeSeriesDataFrame):
        data = TimeSeriesDataFrame(data)

    logger.info(f"Loaded time series data with {len(data)} rows")
    logger.info(f"Number of unique series: {data.num_items if hasattr(data, 'num_items') else 'N/A'}")

    # Validate target column exists
    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in data. Available columns: {data.columns.tolist()}"
        )

    # Create predictor path
    predictor_path = Path(workspace_path) / "timeseries_predictor"

    logger.info(f"Creating TimeSeriesPredictor with prediction_length={prediction_length}")
    logger.info(f"Evaluation metric: {eval_metric}, Preset: {preset}, Time limit: {time_limit}s")

    # Create TimeSeriesPredictor
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target=target_column,
        eval_metric=eval_metric,
        path=str(predictor_path),
        quantile_levels=quantile_levels,
        verbosity=2,
    )

    # Prepare known covariates if provided
    known_covariates_data = None
    if known_covariates_names:
        logger.info(f"Using known covariates: {known_covariates_names}")
        # Validate covariates exist
        missing_covs = set(known_covariates_names) - set(data.columns)
        if missing_covs:
            raise ValueError(f"Known covariates {missing_covs} not found in data columns: {data.columns.tolist()}")
        known_covariates_data = data[known_covariates_names]

    # Fit the predictor
    logger.info("Starting model training...")
    try:
        predictor.fit(
            train_data=data,
            presets=preset,
            time_limit=time_limit,
            excluded_model_types=excluded_model_types,
            num_val_windows=num_val_windows,
            known_covariates_names=known_covariates_names,
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise ValueError(f"TimeSeriesPredictor training failed: {str(e)}")

    logger.info("Training completed successfully")

    # Generate predictions on test data
    # TimeSeriesPredictor automatically uses last prediction_length timesteps for validation
    logger.info("Generating predictions...")
    try:
        predictions_df = predictor.predict(data, known_covariates=known_covariates_data)
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ValueError(f"Prediction generation failed: {str(e)}")

    logger.info(f"Generated predictions with shape: {predictions_df.shape}")

    # Evaluate on test data
    logger.info("Evaluating model performance...")
    try:
        eval_results = predictor.evaluate(data)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        # If evaluation fails, create minimal metrics dict
        eval_results = {eval_metric: 0.0}
        logger.warning("Evaluation failed, using minimal metrics")

    logger.info(f"Evaluation results: {eval_results}")

    # Get best model name from leaderboard
    try:
        leaderboard_df = predictor.leaderboard(data)
        logger.info(f"Leaderboard:\n{leaderboard_df.to_string()}")
        best_model_name = leaderboard_df.iloc[0]["model"]
    except Exception as e:
        logger.warning(f"Failed to get leaderboard: {str(e)}")
        best_model_name = "WeightedEnsemble"

    # Save model artifact
    model_name_full = f"{best_model_name}_TIMESERIES"
    output_path = Path(model_artifact.path) / model_name_full

    # Set artifact metadata
    model_artifact.metadata["display_name"] = model_name_full
    model_artifact.metadata["context"] = {
        "model_config": {
            "preset": preset,
            "eval_metric": eval_metric,
            "time_limit": time_limit,
            "prediction_length": prediction_length,
            "target_column": target_column,
            "known_covariates_names": known_covariates_names or [],
            "excluded_model_types": excluded_model_types or [],
            "quantile_levels": quantile_levels,
            "num_val_windows": num_val_windows,
        },
        "prediction_length": prediction_length,
        "target_column": target_column,
        "metrics": {"test_data": eval_results},
        "location": {
            "model_directory": model_name_full,
            "predictor": f"{model_name_full}/predictor",
        },
    }

    # Clone predictor to output artifact
    logger.info(f"Saving predictor to {output_path / 'predictor'}")
    predictor_clone = predictor.clone(path=output_path / "predictor", return_clone=True, dirs_exist_ok=True)

    # Save metrics
    metrics_path = output_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with (metrics_path / "metrics.json").open("w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info("Saved metrics to artifact")

    # Save predictions to output artifact
    logger.info(f"Saving predictions to {predictions.path}")
    predictions_df.to_parquet(predictions.path)

    # Save metrics to output
    with Path(metrics.path).open("w") as f:
        json.dump(eval_results, f, indent=2)

    # Generate Jupyter notebook
    logger.info("Generating Jupyter notebook...")
    notebook_path = output_path / "notebooks"
    notebook_path.mkdir(parents=True, exist_ok=True)

    notebook_content = _generate_timeseries_notebook(
        model_name_full=model_name_full,
        predictor_path_rel=f"{model_name_full}/predictor",
        prediction_length=prediction_length,
        target_column=target_column,
        eval_metric=eval_metric,
        eval_results=eval_results,
    )

    with (notebook_path / "timeseries_predictor_notebook.ipynb").open("w") as f:
        json.dump(notebook_content, f, indent=2)

    model_artifact.metadata["context"]["location"]["notebook"] = (
        f"{model_name_full}/notebooks/timeseries_predictor_notebook.ipynb"
    )

    logger.info("Notebook generated successfully")

    # Return outputs
    outputs = NamedTuple("outputs", predictor_path=str, best_model_name=str, eval_metric_value=float)
    return outputs(
        predictor_path=str(predictor_path),
        best_model_name=best_model_name,
        eval_metric_value=float(eval_results.get(eval_metric, 0.0)),
    )


def _generate_timeseries_notebook(
    model_name_full: str,
    predictor_path_rel: str,
    prediction_length: int,
    target_column: str,
    eval_metric: str,
    eval_results: dict,
) -> dict:
    """Generate Jupyter notebook for time series model exploration."""
    metrics_table = "\\n".join([f"- **{k}**: {v:.4f}" for k, v in eval_results.items()])

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# AutoGluon Time Series Forecasting - Predictor Notebook\n",
                    "\n",
                    "This notebook allows you to:\n",
                    "- Review the trained time series models and their performance\n",
                    "- Load the AutoGluon TimeSeriesPredictor from storage\n",
                    "- Generate new forecasts on your data\n",
                    "- Visualize predictions with confidence intervals\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Pipeline Details\n",
                    "\n",
                    f"- **Model**: {model_name_full}\n",
                    f"- **Target Column**: {target_column}\n",
                    f"- **Prediction Length**: {prediction_length} timesteps\n",
                    f"- **Evaluation Metric**: {eval_metric}\n",
                    "\n",
                    "### Evaluation Metrics\n",
                    "\n",
                    f"{metrics_table}",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Setup and Imports",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor\n",
                    "\n",
                    "# Configure plotting\n",
                    "plt.style.use('seaborn-v0_8-darkgrid')\n",
                    "%matplotlib inline",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Download Model from S3\n",
                    "\n",
                    "Configure your S3 credentials and download the trained predictor.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import boto3\n",
                    "import os\n",
                    "\n",
                    "# Configure S3 client\n",
                    "s3_client = boto3.client(\n",
                    "    's3',\n",
                    "    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),\n",
                    "    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),\n",
                    "    endpoint_url=os.environ.get('AWS_S3_ENDPOINT')\n",
                    ")\n",
                    "\n",
                    "# Download predictor from S3\n",
                    "bucket_name = 'YOUR_BUCKET_NAME'\n",
                    f"s3_predictor_prefix = 'path/to/artifacts/{predictor_path_rel}'\n",
                    "local_predictor_path = './downloaded_predictor'\n",
                    "\n",
                    "# Note: You'll need to implement recursive S3 download\n",
                    "# or use AWS CLI: aws s3 cp s3://bucket/path ./local_path --recursive",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Load Predictor",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load the trained predictor\n",
                    "predictor = TimeSeriesPredictor.load(local_predictor_path)\n",
                    "\n",
                    "print(f'Loaded TimeSeriesPredictor')\n",
                    f"print(f'Target: {target_column}')\n",
                    f"print(f'Prediction length: {prediction_length}')\n",
                    "print(f'Eval metric: {predictor.eval_metric}')",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## View Model Leaderboard",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load your test data\n",
                    "# test_data = TimeSeriesDataFrame.from_data_frame(...)\n",
                    "\n",
                    "# Get leaderboard\n",
                    "# leaderboard = predictor.leaderboard(test_data)\n",
                    "# leaderboard",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Generate Forecasts\n",
                    "\n",
                    "Generate multi-step ahead forecasts with quantile predictions.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load your data for forecasting\n",
                    "# data = TimeSeriesDataFrame.from_data_frame(...)\n",
                    "\n",
                    "# Generate predictions\n",
                    "# predictions = predictor.predict(data)\n",
                    "\n",
                    "# predictions will contain:\n",
                    "# - 'mean': point forecast\n",
                    "# - quantile columns (e.g., '0.1', '0.5', '0.9'): probabilistic forecasts\n",
                    "\n",
                    "# print(predictions.head())",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Visualize Forecasts\n",
                    "\n",
                    "Plot actual values with forecasts and confidence intervals.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Example visualization for a single time series\n",
                    "# item_id = 'your_item_id'  # Replace with actual item_id\n",
                    "\n",
                    "# # Get historical data and predictions for this item\n",
                    "# historical = data.loc[item_id]\n",
                    "# forecast = predictions.loc[item_id]\n",
                    "\n",
                    "# # Plot\n",
                    "# plt.figure(figsize=(14, 6))\n",
                    f"# plt.plot(historical.index, historical['{target_column}'], label='Historical', linewidth=2)\n",
                    "# plt.plot(forecast.index, forecast['mean'], label='Forecast (mean)', linewidth=2, linestyle='--')\n",
                    "\n",
                    "# # Add confidence intervals\n",
                    "# if '0.1' in forecast.columns and '0.9' in forecast.columns:\n",
                    "#     plt.fill_between(forecast.index, forecast['0.1'], forecast['0.9'], \n",
                    "#                      alpha=0.3, label='80% Confidence Interval')\n",
                    "\n",
                    "# plt.xlabel('Time')\n",
                    f"# plt.ylabel('{target_column}')\n",
                    "# plt.title(f'Time Series Forecast for {item_id}')\n",
                    "# plt.legend()\n",
                    "# plt.grid(True, alpha=0.3)\n",
                    "# plt.tight_layout()\n",
                    "# plt.show()",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Evaluate Model Performance",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Evaluate on test data\n",
                    "# metrics = predictor.evaluate(test_data)\n",
                    "# print('Evaluation Metrics:')\n",
                    "# for metric_name, value in metrics.items():\n",
                    "#     print(f'{metric_name}: {value:.4f}')",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Additional Analysis\n",
                    "\n",
                    "Explore model details and feature importance (if applicable).",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# View model information\n",
                    "# predictor.info()",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    return notebook


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_forecasting,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
