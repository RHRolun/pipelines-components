from typing import List, NamedTuple, Optional

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
    packages_to_install=[
        "autogluon.timeseries==1.5.0",
        "pandas==2.2.3",
    ],
)
def timeseries_models_selection(
    target: str,
    id_column: str,
    timestamp_column: str,
    train_data_path: str,
    test_data: dsl.Input[dsl.Dataset],
    top_n: int,
    workspace_path: str,
    prediction_length: int = 1,
    known_covariates_names: Optional[List[str]] = None,
) -> NamedTuple(
    "outputs",
    top_models=List[str],
    predictor_path=str,
    eval_metric_name=str,
    model_config=dict,
):
    """Train and select top N AutoGluon timeseries models based on leaderboard.

    This component trains multiple AutoGluon TimeSeries models using TimeSeriesPredictor
    on the selection training data, evaluates them on the test set, and selects the
    top N performers based on the leaderboard ranking.

    The TimeSeriesPredictor automatically trains various model types (DeepAR, TFT,
    ARIMA, ETS, Theta, etc.) and ranks them by the evaluation metric. This component
    selects the top N models from the leaderboard for refitting on the full dataset.

    Args:
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id).
        timestamp_column: Name of the timestamp/datetime column.
        train_data_path: Path to the selection training CSV file.
        test_data: Test dataset artifact for evaluation.
        top_n: Number of top models to select for refitting.
        workspace_path: Workspace directory where predictor will be saved.
        prediction_length: Forecast horizon (number of timesteps).
        known_covariates_names: Optional list of known covariate column names.

    Returns:
        NamedTuple: top_models list, predictor_path, eval_metric_name, model_config.
    """
    import logging
    from pathlib import Path

    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    logger = logging.getLogger(__name__)

    # Set constants
    DEFAULT_PRESETS = "medium_quality"
    DEFAULT_EVAL_METRIC = "MASE"
    DEFAULT_TIME_LIMIT = 60 * 60  # 60 * 60 = 3600 seconds = 1 hour

    # Load training data
    logger.info("Loading training data from %s", train_data_path)
    train_df = pd.read_csv(train_data_path)
    logger.info(f"Loaded {len(train_df)} training rows")

    # Load test data
    logger.info("Loading test data from %s", test_data.path)
    test_df = pd.read_csv(test_data.path)
    logger.info(f"Loaded {len(test_df)} test rows")

    # Convert training data to TimeSeriesDataFrame
    logger.info(
        f"Converting data to TimeSeriesDataFrame with id_column={id_column}, timestamp_column={timestamp_column}"
    )
    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    logger.info(f"Created TimeSeriesDataFrame with {len(train_ts)} rows, {train_ts.num_items} items")

    # Convert test data to TimeSeriesDataFrame
    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    # Create predictor path in workspace
    predictor_path = Path(workspace_path) / "timeseries_predictor"

    logger.info(f"Creating TimeSeriesPredictor with prediction_length={prediction_length}")

    eval_metric = DEFAULT_EVAL_METRIC
    # Create TimeSeriesPredictor
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        target=target,
        eval_metric=eval_metric,
        path=str(predictor_path),
        verbosity=2,
    )

    # Fit the predictor
    logger.info("Starting model training...")
    try:
        predictor.fit(
            train_data=train_ts,
            presets=DEFAULT_PRESETS,
            time_limit=DEFAULT_TIME_LIMIT,
            known_covariates_names=known_covariates_names,
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise ValueError(f"TimeSeriesPredictor training failed: {str(e)}")

    logger.info("Training completed successfully")

    # Get leaderboard to select top N models
    logger.info("Generating leaderboard...")
    try:
        leaderboard = predictor.leaderboard(test_ts)
        logger.info(f"Leaderboard:\n{leaderboard.to_string()}")
    except Exception as e:
        logger.error(f"Leaderboard generation failed: {str(e)}")
        raise ValueError(f"Failed to generate leaderboard: {str(e)}")

    # Select top N models from leaderboard
    top_models = leaderboard.head(top_n)["model"].values.tolist()
    logger.info(f"Top {top_n} models selected from leaderboard: {top_models}")
    logger.info(f"Best model: {top_models[0]} with {eval_metric}={leaderboard.iloc[0][eval_metric]}")

    # Create model config
    model_config = {
        "prediction_length": prediction_length,
        "eval_metric": eval_metric,
        "target": target,
        "id_column": id_column,
        "timestamp_column": timestamp_column,
        "presets": DEFAULT_PRESETS,
        "time_limit": DEFAULT_TIME_LIMIT,
        "known_covariates_names": known_covariates_names or [],
        "num_models_trained": len(leaderboard),
    }

    outputs = NamedTuple(
        "outputs",
        top_models=List[str],
        predictor_path=str,
        eval_metric_name=str,
        model_config=dict,
    )
    return outputs(
        top_models=top_models,
        predictor_path=str(predictor_path),
        eval_metric_name=eval_metric,
        model_config=model_config,
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_models_selection,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
