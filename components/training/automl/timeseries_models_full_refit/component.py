from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
)
def timeseries_models_full_refit(
    model_name: str,
    test_dataset: dsl.Input[dsl.Dataset],
    predictor_path: str,
    sampling_config: dict,
    split_config: dict,
    model_config: dict,
    pipeline_name: str,
    run_id: str,
    extra_train_data_path: str,
    model_artifact: dsl.Output[dsl.Model],
):
    """Refit a single AutoGluon timeseries model on full training data.

    This component takes a model selected during the selection phase and
    refits it on the full training dataset (selection + extra train data)
    for improved performance. The refitted model is optimized and saved
    for deployment.

    The component uses a simplified/mocked implementation for demonstration.

    Args:
        model_name: Name of the model to refit.
        test_dataset: Test dataset artifact for evaluation.
        predictor_path: Path to the predictor from selection phase.
        sampling_config: Configuration used for data sampling.
        split_config: Configuration used for data splitting.
        model_config: Model configuration from selection phase.
        pipeline_name: Pipeline name for metadata.
        run_id: Pipeline run ID for metadata.
        extra_train_data_path: Path to extra training data CSV.
        model_artifact: Output artifact for the refitted model.
    """
    import json
    import logging
    from pathlib import Path

    import pandas as pd

    logger = logging.getLogger(__name__)

    # SIMPLIFIED/MOCKED IMPLEMENTATION
    logger.info(f"Refitting model: {model_name}")
    logger.info(f"Predictor path: {predictor_path}")
    logger.info(f"Extra train data: {extra_train_data_path}")

    # Load extra training data
    extra_train_df = pd.read_csv(extra_train_data_path)
    logger.info(f"Loaded {len(extra_train_df)} extra training rows")

    # Load test data
    test_df = pd.read_csv(test_dataset.path)
    logger.info(f"Loaded {len(test_df)} test rows")

    # Mock: Simulate refitting on full data
    logger.info(f"Refitting {model_name} on full training dataset...")

    # Mock: Improved performance after refitting
    mock_metrics = {
        "MASE": 0.72,  # Better than selection phase
        "MAPE": 12.5,
        "SMAPE": 11.8,
        "MSE": 45.2,
        "RMSE": 6.72,
    }

    logger.info(f"Refitting completed. Metrics: {mock_metrics}")

    # Create model output directory
    model_name_full = f"{model_name}_FULL"
    output_path = Path(model_artifact.path) / model_name_full
    output_path.mkdir(parents=True, exist_ok=True)

    # Save mock predictor
    predictor_output = output_path / "predictor"
    predictor_output.mkdir(parents=True, exist_ok=True)

    predictor_metadata = {
        "model_name": model_name_full,
        "base_model": model_name,
        "refitted": True,
        "prediction_length": model_config.get("prediction_length", 1),
        "eval_metric": model_config.get("eval_metric", "MASE"),
        "target": model_config.get("target"),
    }

    with open(predictor_output / "predictor_metadata.json", "w") as f:
        json.dump(predictor_metadata, f, indent=2)

    logger.info(f"Saved predictor to {predictor_output}")

    # Save metrics
    metrics_path = output_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    with open(metrics_path / "metrics.json", "w") as f:
        json.dump(mock_metrics, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")

    # Set artifact metadata
    model_artifact.metadata["display_name"] = model_name_full
    model_artifact.metadata["context"] = {
        "model_config": model_config,
        "sampling_config": sampling_config,
        "split_config": split_config,
        "metrics": {"test_data": mock_metrics},
        "location": {
            "model_directory": model_name_full,
            "predictor": f"{model_name_full}/predictor",
            "metrics": f"{model_name_full}/metrics",
        },
        "pipeline_info": {
            "pipeline_name": pipeline_name,
            "run_id": run_id,
        },
    }

    logger.info(f"Model artifact created: {model_name_full}")


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_models_full_refit,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
