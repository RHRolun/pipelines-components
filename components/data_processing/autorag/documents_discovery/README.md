<!-- markdownlint-disable MD013 -->
# Documents sampling 📄

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Lists documents from S3 and performs document sampling.
It writes a YAML manifest (descriptor) describing the sampled set so that downstream
components (e.g. text extraction) can fetch only the documents they need.

The documents sampling component is the initial step in the AutoRAG pipeline workflow.
It lists objects in the given S3 bucket/prefix, filters by supported document formats,
and applies sampling (e.g. test-data–driven with a size limit). It doesn't download
or copy document bytes; it only produces a manifest file. The component integrates
with S3 via environment-based credentials (e.g. RHOAI Connections) using ibm_boto3.

## Inputs 📥

| Parameter                | Type                      | Default   | Description                                                         |
|--------------------------|---------------------------|-----------|---------------------------------------------------------------------|
| `input_data_bucket_name` | `str`                     | Mandatory | Name of the S3 bucket containing input data.                        |
| `input_data_path`        | `str`                     | Mandatory | Path prefix for listing objects (folder with input documents).      |
| `test_data`              | `dsl.Input[dsl.Artifact]` | `None`    | Optional input artifact containing test data for document sampling. |
| `sampling_config`        | `dict`                    | `None`    | Optional dictionary with sampling configuration.                    |

### Input data

To access the input data stored in an S3-compatible storage, the component requires the following environment variables to be available at runtime:

| Environment variable name | Description                                         |
|--------------------------|-----------------------------------------------------|
| `AWS_ACCESS_KEY_ID`      | access key used to authenticate with the S3 service |
| `AWS_SECRET_ACCESS_KEY`  | secret key used to authenticate with the S3 service |
| `AWS_S3_ENDPOINT`        | endpoint URL of the S3 instance                     |
| `AWS_REGION`             | region in which the S3 instance is deployed         |

### Sampling Configuration

The `sampling_config` dictionary supports test data driven sampling:

- Sample documents referenced in test data
- Add noise documents up to 1GB limit (in-memory)

## Outputs 📤

| Output              | Type           | Description                                               |
|---------------------|----------------|-----------------------------------------------------------|
| `sampled_documents` | `dsl.Artifact` | Artifact containing `sampled_documents_descriptor.yaml`   |  

### Sampled documents descriptor (YAML)

The artifact is a directory containing a single file: **`sampled_documents_descriptor.yaml`**.
It describes the sampled set and S3 locations so downstream components can fetch documents on demand.

| Field               | Description                                              |
|---------------------|----------------------------------------------------------|
| `bucket`            | S3 bucket name.                                          |
| `prefix`            | Path prefix used when listing objects.                   |
| `documents`         | List of entries, each with:                              |
| → `key`             | S3 object key (full path in bucket).                     |
| → `size_bytes`      | File size in bytes.                                      |
| `total_size_bytes`  | Sum of `size_bytes` for all documents.                   |
| `count`             | Number of documents.                                     |

## Usage Examples 💡

### Basic Usage

```python
from kfp import dsl
from kfp_components.components.data_processing.autorag.documents_sampling import documents_sampling

@dsl.pipeline(name="document-loading-pipeline")
def my_pipeline():
    """Example pipeline demonstrating document loading."""
    load_task = documents_sampling(
        input_data_bucket_name="s3-documents-bucket",
        input_data_path="documents-path"
    )
    return load_task
```

### With Test Data Sampling

```python
@dsl.pipeline(name="document-loading-with-sampling-pipeline")
def my_pipeline(test_data):
    """Example pipeline with document sampling."""
    load_task = documents_sampling(
        input_data_bucket_name="s3-documents-bucket",
        input_data_path="documents-path",
        test_data=test_data,
        sampling_config={
            "method": "test_data_driven",
            "max_size_gb": 1.0
        }
    )
    return load_task
```

## Supported Document Types 📋

- **PDF** (`.pdf`) - Portable Document Format
- **DOCX** (`.docx`) - Microsoft Word documents
- **PPTX** (`.pptx`) - Microsoft PowerPoint presentations
- **Markdown** (`.md`) - Markdown files
- **HTML** (`.html`) - HTML documents
- **Plain text** (`.txt`) - Text files

## Notes 📝

- **No download**: This component does not download or copy document bytes; it only lists S3 and writes the descriptor YAML.
- **Document sampling**: Sampling is integrated (e.g. test-data–driven, up to 1GB total size); selected keys are written in the descriptor.
- **Downstream fetch**: Use the descriptor with the text_extraction component (or similar) to fetch and process documents from S3.
- **Credentials**: S3 access requires `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, and `AWS_REGION` at runtime.

## Metadata 🗂️

- **Name**: documents_sampling
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: RHOAI Connections API, Version: >=1.0.0
    - Name: ai4rag, Version: >=1.0.0
- **Tags**:
  - data-processing
  - autorag
  - document-loading
- **Last Verified**: 2026-01-23 10:29:35+00:00

## Additional Resources 📚

- **AutoRAG Documentation**: See AutoRAG pipeline documentation for comprehensive information
- **Issue Tracker**: [GitHub Issues](https://github.com/kubeflow/pipelines-components/issues)
