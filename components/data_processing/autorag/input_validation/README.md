# Input Validation

> **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Validates parameters passed to the documents RAG optimization pipeline. This component is intended
as the first step of the pipeline: it checks required string parameters, `optimization_metric`,
`optimization_max_rag_patterns`, and optional lists.

## Inputs

| Parameter                            | Type                     | Default           | Description                                                          |
|--------------------------------------|--------------------------|-------------------|----------------------------------------------------------------------|
| `test_data_secret_name`              | `str`                    | Mandatory         | Kubernetes secret name for test data S3 access.                      |
| `test_data_bucket_name`              | `str`                    | Mandatory         | S3 bucket for the test data file.                                    |
| `test_data_key`                      | `str`                    | Mandatory         | S3 object key to the test data JSON file (must end with `.json`).    |
| `input_data_secret_name`             | `str`                    | Mandatory         | Kubernetes secret name for input data S3 access.                     |
| `input_data_bucket_name`             | `str`                    | Mandatory         | S3 bucket for the input documents.                                   |
| `input_data_key`                     | `str`                    | Mandatory         | S3 object key (path) for the input documents.                        |
| `llama_stack_secret_name`            | `str`                    | Mandatory         | Kubernetes secret name for llama-stack API.                          |
| `optimization_metric`                | `str`                    | `"faithfulness"`  | One of: `faithfulness`, `answer_correctness`, `context_correctness`. |
| `optimization_max_rag_patterns`      | `int`                    | `8`               | Max RAG patterns to generate (1–20).                                 |
| `embeddings_models`                  | `Optional[List[str]]`    | `None`            | Optional list of embedding model identifiers.                        |
| `generation_models`                  | `Optional[List[str]]`    | `None`            | Optional list of generation model identifiers.                       |
| `llama_stack_vector_database_id`     | `Optional[str]`          | `None`            | Optional vector database id for llama-stack.                         |

## Usage

Used as the first step of the documents RAG optimization pipeline so that invalid parameters fail
fast before any S3 or downstream work runs.

```python
from kfp import dsl
from kfp_components.components.data_processing.autorag.input_validation import input_validation

@dsl.pipeline(name="autorag-pipeline")
def pipeline(
    test_data_secret_name: str,
    test_data_bucket_name: str,
    test_data_key: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    input_data_key: str,
    llama_stack_secret_name: str,
):
    validation_task = input_validation(
        test_data_secret_name=test_data_secret_name,
        test_data_bucket_name=test_data_bucket_name,
        test_data_key=test_data_key,
        input_data_secret_name=input_data_secret_name,
        input_data_bucket_name=input_data_bucket_name,
        input_data_key=input_data_key,
        llama_stack_secret_name=llama_stack_secret_name,
    )
```

## Raises

- **ValueError** — If any parameter fails validation (e.g. empty required string, invalid
  `optimization_metric`, or `optimization_max_rag_patterns` out of range).
