from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas"],
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
    eval_metric: str = "MASE",
) -> NamedTuple(
    "outputs",
    top_models=List[str],
    predictor_path=str,
    eval_metric_name=str,
    model_config=dict,
):
    """Train and select top N AutoGluon timeseries models.

    This component trains multiple AutoGluon TimeSeries models on the
    selection training data, evaluates them on the test set, and selects
    the top N performers based on the specified evaluation metric.

    The component uses a simplified/mocked implementation for demonstration.

    Args:
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id).
        timestamp_column: Name of the timestamp/datetime column.
        train_data_path: Path to the selection training CSV file.
        test_data: Test dataset artifact for evaluation.
        top_n: Number of top models to select for refitting.
        workspace_path: Workspace directory where predictor will be saved.
        prediction_length: Forecast horizon (number of timesteps).
        eval_metric: Evaluation metric (e.g., "MASE", "MAPE", "SMAPE").

    Returns:
        NamedTuple: top_models list, predictor_path, eval_metric_name, model_config.
    """
    import json
    import logging
    from pathlib import Path

    import pandas as pd

    logger = logging.getLogger(__name__)

    # SIMPLIFIED/MOCKED IMPLEMENTATION
    logger.info("Loading training data from %s", train_data_path)
    train_df = pd.read_csv(train_data_path)
    logger.info(f"Loaded {len(train_df)} training rows")

    logger.info("Loading test data from %s", test_data.path)
    test_df = pd.read_csv(test_data.path)
    logger.info(f"Loaded {len(test_df)} test rows")

    # Create predictor path in workspace
    predictor_path = Path(workspace_path) / "timeseries_predictor"
    predictor_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training models with prediction_length={prediction_length}, eval_metric={eval_metric}")

    # Mock: Simulate training multiple models
    all_models = [
        "DeepAR",
        "TemporalFusionTransformer",
        "AutoARIMA",
        "AutoETS",
        "Theta",
        "SeasonalNaive",
        "DirectTabular",
        "RecursiveTabular",
    ]

    # Mock: Simulate model scores (lower is better for MASE)
    model_scores = {
        "DeepAR": 0.85,
        "TemporalFusionTransformer": 0.92,
        "AutoARIMA": 0.78,
        "AutoETS": 0.81,
        "Theta": 0.88,
        "SeasonalNaive": 1.15,
        "DirectTabular": 0.95,
        "RecursiveTabular": 0.99,
    }

    # Select top N models
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
    top_models = [model for model, score in sorted_models[:top_n]]

    logger.info(f"Top {top_n} models selected: {top_models}")
    logger.info(f"Best model: {top_models[0]} with {eval_metric}={sorted_models[0][1]}")

    # Save mock predictor metadata
    predictor_metadata = {
        "target": target,
        "id_column": id_column,
        "timestamp_column": timestamp_column,
        "prediction_length": prediction_length,
        "eval_metric": eval_metric,
        "trained_models": all_models,
        "top_models": top_models,
        "model_scores": model_scores,
    }

    with open(predictor_path / "predictor_metadata.json", "w") as f:
        json.dump(predictor_metadata, f, indent=2)

    logger.info(f"Saved predictor metadata to {predictor_path}")

    # Create model config
    model_config = {
        "prediction_length": prediction_length,
        "eval_metric": eval_metric,
        "target": target,
        "id_column": id_column,
        "timestamp_column": timestamp_column,
        "num_models_trained": len(all_models),
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
