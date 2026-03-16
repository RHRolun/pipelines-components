import pathlib

from kfp import dsl

_NOTEBOOKS_DIR = str(pathlib.Path(__file__).parent / "notebook_templates")


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    # noqa: E501
    packages_to_install=[
        "autogluon.timeseries==1.5.0",
        "pandas==2.2.3",
    ],
    embedded_artifact_path=_NOTEBOOKS_DIR,
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
    notebooks: dsl.EmbeddedInput[dsl.Dataset],
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
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    logger = logging.getLogger(__name__)

    logger.info(f"Processing model: {model_name}")
    logger.info(f"Predictor path: {predictor_path}")
    logger.info(f"Extra train data: {extra_train_data_path}")

    # Load the predictor from selection phase
    logger.info(f"Loading predictor from {predictor_path}")
    try:
        original_predictor = TimeSeriesPredictor.load(predictor_path)
        logger.info(f"Predictor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load predictor: {str(e)}")
        raise ValueError(f"Could not load predictor from {predictor_path}: {str(e)}")


    # Load extra training data
    extra_train_df = pd.read_csv(extra_train_data_path)
    logger.info(f"Loaded {len(extra_train_df)} extra training rows")

    # Load test data
    test_df = pd.read_csv(test_dataset.path)
    logger.info(f"Loaded {len(test_df)} test rows")

    # Convert test data to TimeSeriesDataFrame for evaluation
    id_column = model_config.get("id_column")
    timestamp_column = model_config.get("timestamp_column")

    logger.info(
        f"Converting test data to TimeSeriesDataFrame with id_column={id_column}, timestamp_column={timestamp_column}"
    )
    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )


    # clone the predictor to the output artifact path and delete unnecessary models
    predictor = original_predictor.clone(path=output_path / "predictor", return_clone=True, dirs_exist_ok=True)
    predictor.delete_models(models_to_keep=[model_name])

    # MOCK: Skip actual refitting (commented out for demonstration)
    # TODO: Add fit here
    logger.info(f"Skipping refit for {model_name} (mocked implementation)")

    # Evaluate the model using predictor.evaluate
    logger.info(f"Evaluating model {model_name} on test data...")
    try:
        metrics = predictor.evaluate(test_ts)
        logger.info(f"Evaluation completed. Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise ValueError(f"Failed to evaluate model: {str(e)}")

    # Create model output directory
    model_name_full = f"{model_name}_FULL"
    output_path = Path(model_artifact.path) / model_name_full
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the predictor with the selected model
    predictor_output = output_path / "predictor"

    logger.info(f"Saving predictor to {predictor_output}")
    try:
        # Save the predictor to the output path
        predictor.set_model_best(model=model_name_full, save_trainer=True)
        predictor.save_space()
        logger.info(f"Predictor saved successfully to {predictor_output}")
    except Exception as e:
        logger.error(f"Failed to save predictor: {str(e)}")
        raise ValueError(f"Could not save predictor to {predictor_output}: {str(e)}")

    # Save additional metadata about the selected model
    predictor_metadata = {
        "model_name": model_name_full,
        "base_model": model_name,
        "selected_model": model_name,
        "refitted": False,  # Currently using mocked refit
        "prediction_length": model_config.get("prediction_length", 1),
        "eval_metric": model_config.get("eval_metric", "MASE"),
        "target": model_config.get("target"),
        "id_column": model_config.get("id_column"),
        "timestamp_column": model_config.get("timestamp_column"),
    }

    with open(predictor_output / "predictor_metadata.json", "w") as f:
        json.dump(predictor_metadata, f, indent=2)

    logger.info(f"Saved predictor metadata to {predictor_output / 'predictor_metadata.json'}")

    # Save metrics
    metrics_path = output_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    # Convert metrics to JSON-serializable format
    metrics_dict = {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}

    with open(metrics_path / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info(f"Saved metrics to {metrics_path}")

    # Notebook generation

    notebook_file = "timeseries_notebook.ipynb"

    import os

    with open(os.path.join(notebooks.path, notebook_file), "r", encoding="utf-8") as f:
        notebook = json.load(f)
    notebook_path = output_path / "notebooks"
    notebook_path.mkdir(parents=True, exist_ok=True)
    with (notebook_path / "automl_predictor_notebook.ipynb").open("w", encoding="utf-8") as f:
        json.dump(notebook, f)

    # Set artifact metadata
    model_artifact.metadata["display_name"] = model_name_full
    model_artifact.metadata["context"] = {
        "model_config": model_config,
        "sampling_config": sampling_config,
        "split_config": split_config,
        "metrics": {"test_data": metrics_dict},
        "location": {
            "model_directory": model_name_full,
            "predictor": f"{model_name_full}/predictor",
            "metrics": f"{model_name_full}/metrics",
            "notebooks": f"{model_name_full}/notebooks",
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
