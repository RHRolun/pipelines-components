# Timeseries Models Full Refit ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Refit a single AutoGluon timeseries model on full training data.

This component takes a model selected during the selection phase and refits it on the full training dataset (selection +
extra train data) for improved performance. The refitted model is optimized and saved for deployment.

The component uses a simplified/mocked implementation for demonstration.

## Inputs 📥

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | `str` | `None` | Name of the model to refit. |
| `test_dataset` | `dsl.Input[dsl.Dataset]` | `None` | Test dataset artifact for evaluation. |
| `predictor_path` | `str` | `None` | Path to the predictor from selection phase. |
| `sampling_config` | `dict` | `None` | Configuration used for data sampling. |
| `split_config` | `dict` | `None` | Configuration used for data splitting. |
| `model_config` | `dict` | `None` | Model configuration from selection phase. |
| `pipeline_name` | `str` | `None` | Pipeline name for metadata. |
| `run_id` | `str` | `None` | Pipeline run ID for metadata. |
| `extra_train_data_path` | `str` | `None` | Path to extra training data CSV. |
| `model_artifact` | `dsl.Output[dsl.Model]` | `None` | Output artifact for the refitted model. |

## Metadata 🗂️

- **Name**: timeseries_models_full_refit
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
  - timeseries
  - automl
  - model-refit
- **Last Verified**: 2026-03-13 13:51:54+00:00
- **Owners**:
  - Approvers: None
  - Reviewers: None
