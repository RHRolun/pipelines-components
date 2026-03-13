from typing import List, Optional

from kfp import dsl
from kfp_components.components.data_processing.automl.timeseries_data_loader import timeseries_data_loader
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import leaderboard_evaluation
from kfp_components.components.training.automl.timeseries_models_full_refit import timeseries_models_full_refit
from kfp_components.components.training.automl.timeseries_models_selection import timeseries_models_selection


@dsl.pipeline(
    name="autogluon-timeseries-training-pipeline",
    description=(
        "End-to-end AutoGluon time series forecasting pipeline. Loads time series data from S3 in "
        "TimeSeriesDataFrame format (item_id, timestamp, target), trains multiple AutoGluon TimeSeries models "
        "(local statistical and global deep learning), ranks them by the chosen eval metric, and produces a "
        "leaderboard and the top N trained predictors for deployment. Supports optional known covariates and "
        "configurable prediction_length."
    ),
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size="1Gi",  # TODO: change to recommended size
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "storageClassName": "gp3-csi",  # or 'gp3', 'fast', etc.
                    "accessModes": ["ReadWriteOnce"],
                }
            ),
        ),
    ),
)
def autogluon_timeseries_training_pipeline(
    train_data_secret_name: str,
    train_data_bucket_name: str,
    train_data_file_key: str,
    target: str,
    id_column: str,
    timestamp_column: str,
    known_covariates_names: Optional[List[str]] = None,
    prediction_length: int = 1,
    top_n: int = 3,
):
    """AutoGluon Time Series Training Pipeline (draft design).

    This pipeline is intended to train AutoGluon time series forecasting models on data loaded from
    S3, evaluate them on a validation split, and output a leaderboard plus the top N trained
    predictors. Components are not yet implemented; this docstring describes the target design
    and parameters.

     **Storage strategy:**

    Training datasets are stored on a PVC workspace (not S3 artifacts) so that all
    pipeline steps sharing the workspace can access them without extra downloads. Only
    the test dataset is written to an S3 artifact (for use by the leaderboard evaluation
    component). The workspace is provisioned via ``PipelineConfig.workspace``.

    **Intended pipeline stages**


    1. **Data Loading & Splitting**: Loads timeseries tabular (CSV) data from an S3-compatible
       object storage bucket using AWS credentials configured via Kubernetes secrets.
       The component samples the data (up to 1GB), then performs a two-stage split:
       *Primary split** (default 80/20): separates a *test set* (20%, written to an
         S3 artifact) from the *train portion* (80%).
         **Secondary split** (default 30/70 of the train portion): produces
         ``models_selection_train_dataset.csv`` (30%, used for model selection) and
         ``extra_train_dataset.csv`` (70%, passed to ``refit_full`` as extra data).
         Both train CSVs are written to the PVC workspace under
         ``{workspace_path}/datasets/``.
        The dataset must be ordered correctly prior to running the pipeline. The data must contain columns
       for item_id, timestamp, and target (and optionally known covariates).

    2.  **Training and model selection on data sample**: Train multiple AutoGluon TimeSeries models (local e.g.
       ARIMA, ETS, Theta; global e.g. DeepAR, TFT) on the training data. Rank by eval_metric
       (e.g. WQL, MASE) on the validation set and select the top N models.

    3. **Model fiting on larger part of input dataset**: Fits each of the top N selected models on the predictor's
       training and validation data, augmented with the *extra train* split via
       ``refit_full(train_data_extra=...)``. This stage runs in parallel (with
       parallelism of 2) to efficiently retrain multiple models. Each refitted model is
       saved with a "_FULL" suffix and optimized for deployment by removing unnecessary
       models and files.

    4. **Leaderboard evaluation**: Aggregate metrics from the trained models and produce an HTML
       leaderboard ranking them by the chosen evaluation metric. Output the top N predictors
       (and optionally metrics, notebook) for deployment.

    **Parameters (data and identity)**

    - **train_data_secret_name**: Kubernetes secret name containing S3 credentials
      (e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION).
    - **train_data_bucket_name**: S3-compatible bucket name containing the time series data file.
    - **train_data_file_key**: S3 object key of the data file (CSV or Parquet). File must include
      columns for item_id, timestamp, and target; optional columns for known covariates.

    **Parameters (TimeSeriesDataFrame schema)**

    - **target**: Name of the column containing the numeric values to forecast (the time series
      values). Corresponds to :attr:`~autogluon.timeseries.TimeSeriesDataFrame` target column.
    - **id_column**: Name of the column that identifies each time series (e.g. product_id, store_id).
      Passed as ``id_column`` when constructing TimeSeriesDataFrame from a DataFrame or path; the
      result uses ``item_id`` as the first index level.
    - **timestamp_column**: Name of the column containing the timestamp/datetime for each observation.
      Passed as ``timestamp_column`` when constructing TimeSeriesDataFrame; the result uses
      ``timestamp`` as the second index level.

    **Parameters (predictor and selection)**

    - **known_covariates_names**: Optional list of column names that are known in advance for all
      steps in the forecast horizon (e.g. holidays, promotions). If provided, the predictor expects
      these columns in the data and at prediction time requires future values in ``known_covariates``.
      See :attr:`~autogluon.timeseries.TimeSeriesPredictor.known_covariates_names`.
    - **prediction_length**: Number of time steps to forecast (horizon length). Required for
      training and evaluation; used for validation split and by the predictor. Positive integer
      (default: 1).
    - **top_n**: Number of top models to select for the leaderboard and output (default: 3). Positive
      integer.

    **Returns**

    Intended outputs (once implemented): leaderboard artifact (HTML), trained model artifacts
    (predictor and metrics for the top N models), and optionally a generated notebook for inference.

    **Raises**

    Expected to raise on: missing or inaccessible S3 file; missing ``target``, ``id_column``, or
    ``timestamp_column`` in the data; invalid ``prediction_length`` or ``top_n``; or failure to build
    TimeSeriesDataFrame (e.g. duplicate item_id/timestamp pairs).

    **Example**

        pipeline = autogluon_timeseries_training_pipeline(
            train_data_secret_name="my-s3-secret",
            train_data_bucket_name="my-bucket",
            train_data_file_key="ts/sales.csv",
            target="sales",
            id_column="product_id",
            timestamp_column="date",
            known_covariates_names=["is_holiday", "promo"],
            prediction_length=14,
            top_n=3,
        )
    """
    # Stage 1: Data Loading & Splitting
    data_loader_task = timeseries_data_loader(
        bucket_name=train_data_bucket_name,
        file_key=train_data_file_key,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    # Configure S3 secret for data loader
    from kfp.kubernetes import use_secret_as_env

    use_secret_as_env(
        data_loader_task,
        secret_name=train_data_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
        optional=True,
    )

    # Stage 2: Model Selection
    # Train multiple models on selection data and select top N performers
    selection_task = timeseries_models_selection(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        train_data_path=data_loader_task.outputs["models_selection_train_data_path"],
        test_data=data_loader_task.outputs["sampled_test_dataset"],
        top_n=top_n,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names,
    )

    # Stage 3: Model Refitting
    # Refit each top model on full training dataset in parallel
    with dsl.ParallelFor(items=selection_task.outputs["top_models"], parallelism=2) as model_name:
        refit_task = timeseries_models_full_refit(
            model_name=model_name,
            test_dataset=data_loader_task.outputs["sampled_test_dataset"],
            predictor_path=selection_task.outputs["predictor_path"],
            sampling_config=data_loader_task.outputs["sample_config"],
            split_config=data_loader_task.outputs["split_config"],
            model_config=selection_task.outputs["model_config"],
            pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
            run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
            extra_train_data_path=data_loader_task.outputs["extra_train_data_path"],
        )

    # Stage 4: Leaderboard Evaluation
    # Generate leaderboard from all refitted models
    leaderboard_evaluation(
        models=dsl.Collected(refit_task.outputs["model_artifact"]),
        eval_metric=selection_task.outputs["eval_metric_name"],
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_training_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
