# Timeseries Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Load and split timeseries data from S3 for AutoGluon training.

This component loads time series data from S3, samples it (up to 1GB), and performs a two-stage split for efficient
AutoGluon training: 1. Primary split (80/20): test set vs train portion 2. Secondary split (30/70 of train):
selection-train vs extra-train

The test set is written to S3 artifact, while train CSVs are written to the PVC workspace for sharing across pipeline
steps.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_key` | `str` | `None` | S3 object key of the CSV file containing time series data. |
| `bucket_name` | `str` | `None` | S3 bucket name containing the file. |
| `workspace_path` | `str` | `None` | PVC workspace directory where train CSVs will be written. |
| `target` | `str` | `None` | Name of the target column to forecast. |
| `id_column` | `str` | `None` | Name of the column identifying each time series (item_id). |
| `timestamp_column` | `str` | `None` | Name of the timestamp/datetime column. |
| `sampled_test_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact for the test split. |
| `selection_train_size` | `float` | `0.3` | Fraction of train portion for model selection (default: 0.3). |

## Outputs 📤

| Name | Type | Description |
|------|------|-------------|
| Output | `NamedTuple('outputs', sample_config=dict, split_config=dict, models_selection_train_data_path=str, extra_train_data_path=str)` | sample_config, split_config, models_selection_train_data_path, extra_train_data_path. |

## Metadata 🗂️

- **Name**: timeseries_data_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - data-processing
  - timeseries
  - automl
  - data-loading
- **Last Verified**: 2026-03-13 13:42:00+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None
