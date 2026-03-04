# AutoRAG Data Processing Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Pipeline to load test data and documents, sample them, and extract text for AutoRAG.

The AutoRAG Data Processing Pipeline prepares input data for the AutoRAG optimization workflow.
It runs on Red Hat OpenShift AI using Kubeflow Pipelines to orchestrate three steps: loading test
data (JSON) from S3, sampling documents from an S3 bucket based on that test data, and extracting
text from the sampled documents using the docling library. The pipeline uses S3-compatible
credentials injected from Kubernetes secrets and produces artifacts (test data, sampled documents
descriptor, extracted text) that can be consumed by downstream pipelines such as the Documents RAG
Optimization Pipeline.

## Inputs 📥

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `test_data_secret_name` | `str` | — | Kubernetes secret name for S3-compatible credentials (test data). Must provide: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION. |
| `input_data_secret_name` | `str` | — | Kubernetes secret name for S3-compatible credentials (input documents). Same env vars as above. |
| `test_data_bucket_name` | `str` | — | S3 (or compatible) bucket that contains the test data JSON file. |
| `test_data_key` | `str` | — | S3 object key to the JSON test data file. |
| `input_data_bucket_name` | `str` | — | S3 (or compatible) bucket containing input documents. |
| `input_data_key` | `str` | — | Path to folder with input documents within the bucket. |
| `sampling_config` | `dict` | — | Sampling configuration dictionary (e.g. test-data–driven sampling, size limits). Can be empty `{}`. |

## Stored artifacts (S3 / results storage) 📁

After pipeline execution, outputs are stored in the pipeline run's artifact location. Layout follows pipeline and component structure:

```text
<pipeline_name>/
└── <run_id>/
    ├── test-data-loader/
    │   └── <task_id>/
    │       └── test_data                    # JSON test data file
    ├── documents-sampling/
    │   └── <task_id>/
    │       └── sampled_documents           # YAML artifact (sampled_documents_descriptor.yaml)
    └── text-extraction/
        └── <task_id>/
            └── extracted_text               # Folder containing markdown files with extracted text
```

- `pipeline_name`: pipeline identifier (e.g. `data_processing_autorag`).
- `run_id`: Kubeflow Pipelines run ID.
- Component folders align with pipeline steps; `<task_id>` is the KFP task ID for that step.

```python
"""Example usage of the data_processing_pipeline."""

from kfp_components.pipelines.data_processing.autorag import data_processing_pipeline


def example_minimal_usage():
    """Minimal example with required parameters."""
    return data_processing_pipeline(
        test_data_secret_name="s3-test-data-secret",
        input_data_secret_name="s3-input-secret",
        test_data_bucket_name="autorag-benchmarks",
        test_data_key="test_data.json",
        input_data_bucket_name="my-documents-bucket",
        input_data_key="documents/",
        sampling_config={},
    )
```

## Metadata 🗂️

- **Name**: autorag
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: docling, Version: >=2.72.0
    - Name: boto3, Version: >=1.42.34
- **Tags**:
  - data_processing
  - text_extraction
  - documents_discovery
- **Last Verified**: 2026-02-04 11:46:16+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
  - Reviewers:
    - LukaszCmielowski


<!-- custom-content -->

## Pipeline Workflow 🔄

The data processing pipeline runs three stages:

1. **Test Data Loading**: Loads the test data JSON file from S3 into a KFP artifact (used for evaluation and for document sampling).
2. **Document Sampling**: Lists documents in the input S3 bucket/prefix, applies sampling (e.g. test-data–driven with optional size limit), and writes a YAML descriptor of the sampled set. Does not download document contents.
3. **Text Extraction**: Reads the descriptor, fetches the listed documents from S3, and extracts text using docling. Outputs markdown files suitable for downstream chunking and embedding.

## Required Parameters ✅

The following parameters are required to run the pipeline:

- `test_data_secret_name` - Kubernetes secret for S3 credentials (test data)
- `input_data_secret_name` - Kubernetes secret for S3 credentials (input documents)
- `test_data_bucket_name` - Bucket containing the test data JSON file
- `test_data_key` - Object key to the test data JSON file
- `input_data_bucket_name` - Bucket containing the input documents
- `input_data_key` - Path to folder with input documents in the bucket
- `sampling_config` - Sampling configuration dict (use `{}` for defaults)

## Components Used 🔧

This pipeline orchestrates the following AutoRAG components:

1. **[Test Data Loader](../../components/data_processing/autorag/test_data_loader/README.md)** -
   Loads test data from a JSON file in S3

2. **[Documents sampling](../../components/data_processing/autorag/documents_sampling/README.md)** -
   Lists documents from S3 and performs sampling; produces a YAML descriptor

3. **[Text Extraction](../../components/data_processing/autorag/text_extraction/README.md)** -
   Fetches documents from S3 and extracts text using docling

## Artifacts 📦

For each pipeline run, the pipeline produces:

- **Test Data Artifact**: JSON file containing the benchmark/test data (e.g. questions and correct answer document IDs).
- **Sampled Documents Artifact**: YAML manifest (`sampled_documents_descriptor.yaml`) with bucket, prefix, and list of sampled document keys for downstream components.
- **Extracted Text Artifact**: Folder of markdown files (one per document) with extracted text, ready for chunking and embedding in RAG optimization pipelines.
