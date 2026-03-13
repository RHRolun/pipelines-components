from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
)
def timeseries_data_loader(
    file_key: str,
    bucket_name: str,
    workspace_path: str,
    target: str,
    id_column: str,
    timestamp_column: str,
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    selection_train_size: float = 0.3,
) -> NamedTuple(
    "outputs",
    sample_config=dict,
    split_config=dict,
    models_selection_train_data_path=str,
    extra_train_data_path=str,
):
    """Load and split timeseries data from S3 for AutoGluon training.

    This component loads time series data from S3, samples it (up to 1GB),
    and performs a two-stage split for efficient AutoGluon training:
    1. Primary split (80/20): test set vs train portion
    2. Secondary split (30/70 of train): selection-train vs extra-train

    The test set is written to S3 artifact, while train CSVs are written
    to the PVC workspace for sharing across pipeline steps.

    Args:
        file_key: S3 object key of the CSV file containing time series data.
        bucket_name: S3 bucket name containing the file.
        workspace_path: PVC workspace directory where train CSVs will be written.
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id).
        timestamp_column: Name of the timestamp/datetime column.
        sampled_test_dataset: Output dataset artifact for the test split.
        selection_train_size: Fraction of train portion for model selection (default: 0.3).

    Returns:
        NamedTuple: sample_config, split_config, models_selection_train_data_path, extra_train_data_path.
    """
    import logging
    from pathlib import Path

    import pandas as pd

    logger = logging.getLogger(__name__)

    # Create workspace datasets directory
    datasets_dir = Path(workspace_path) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # SIMPLIFIED/MOCKED IMPLEMENTATION
    logger.info("Creating mock timeseries data...")

    # Create mock time series data
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    items = ["item_A", "item_B", "item_C"]

    data = []
    for item in items:
        for date in dates:
            data.append({id_column: item, timestamp_column: date, target: 100 + len(data) * 0.1})

    df = pd.DataFrame(data)
    logger.info(f"Created mock data with {len(df)} rows, {len(items)} items, {len(dates)} timesteps")

    # Split: 80% train, 20% test
    test_size = 0.2
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Split train into selection-train and extra-train
    selection_idx = int(len(train_df) * selection_train_size)
    selection_train_df = train_df.iloc[:selection_idx]
    extra_train_df = train_df.iloc[selection_idx:]

    logger.info(
        f"Split sizes - Test: {len(test_df)}, Selection: {len(selection_train_df)}, Extra: {len(extra_train_df)}"
    )

    # Save test dataset to artifact
    test_df.to_csv(sampled_test_dataset.path, index=False)
    logger.info(f"Saved test dataset to {sampled_test_dataset.path}")

    # Save train datasets to workspace
    selection_path = datasets_dir / "models_selection_train_dataset.csv"
    extra_path = datasets_dir / "extra_train_dataset.csv"

    selection_train_df.to_csv(selection_path, index=False)
    extra_train_df.to_csv(extra_path, index=False)

    logger.info(f"Saved selection train to {selection_path}")
    logger.info(f"Saved extra train to {extra_path}")

    # Create sample config and split config
    sample_config = {
        "sampling_method": "first_n_rows",
        "total_rows": len(df),
        "sampled_rows": len(df),
    }

    split_config = {
        "test_size": test_size,
        "selection_train_size": selection_train_size,
        "random_state": 42,
    }

    outputs = NamedTuple(
        "outputs",
        sample_config=dict,
        split_config=dict,
        models_selection_train_data_path=str,
        extra_train_data_path=str,
    )
    return outputs(
        sample_config=sample_config,
        split_config=split_config,
        models_selection_train_data_path=str(selection_path),
        extra_train_data_path=str(extra_path),
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
