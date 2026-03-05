# AutoRAG Documents Indexing Pipeline

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview

Extends the [AutoRAG Data Processing Pipeline](../documents_processing_pipeline/README.md) with a **documents indexing** step. Loads test data, discovers and samples documents, extracts text, then chunks and indexes the extracted text into a vector store (Llama Stack).

## Pipeline workflow

1. **Test data loading** — Loads the test data JSON from S3 (for evaluation and sampling).
2. **Documents discovery** — Lists documents in the input S3 bucket/prefix, applies sampling, writes a descriptor of the sampled set.
3. **Text extraction** — Fetches the listed documents from S3 and extracts text using docling; outputs markdown files.
4. **Documents indexing** — Chunks the extracted text, embeds via Llama Stack, and writes to a vector store collection.

## Inputs

| Parameter                 | Type              | Default   | Description |
|---------------------------|-------------------|-----------|-------------|
| `test_data_secret_name`   | `str`             | —         | Secret with S3 credentials for test data. |
| `input_data_secret_name`  | `str`             | —         | Secret with S3 credentials for input data. |
| `indexing_secret_name`    | `str`             | —         | Secret with Llama Stack credentials (`LLAMA_STACK_CLIENT_BASE_URL`, `LLAMA_STACK_CLIENT_API_KEY`). |
| `input_data_bucket_name`  | `str`             | —         | S3 bucket containing input documents. |
| `input_data_key`          | `str`             | —         | Path to folder with input documents in the bucket. |
| `embedding_model_id`      | `str`             | —         | Embedding model ID for the vector store. |
| `collection_name`         | `str`             | —         | Name of the vector store collection. |
| `embedding_params`        | `dict`            | `{}`      | Dict passed to LSEmbeddingParams. |
| `sampling_enabled`        | `bool`            | `False`   | Whether to enable document sampling. |
| `sampling_max_size`       | `Optional[float]` | `None`    | Max size of sampled documents (GB). |
| `test_data_bucket_name`   | `Optional[str]`   | `None`    | S3 bucket for test data file. |
| `test_data_key`           | `Optional[str]`   | `None`    | S3 object key to test data JSON. |
| `provider_id`             | `Optional[str]`   | `None`    | Optional Llama Stack provider ID. |
| `distance_metric`         | `str`             | `"cosine"`| Vector distance metric. |
| `chunking_method`         | `str`             | `"recursive"` | Chunking method. |
| `chunk_size`              | `int`             | `1024`    | Chunk size in characters. |
| `chunk_overlap`           | `int`             | `0`       | Chunk overlap in characters. |

## Components used

1. [Test data loader](../../../components/data_processing/autorag/test_data_loader/README.md)
2. [Documents discovery](../../../components/data_processing/autorag/documents_discovery/README.md)
3. [Text extraction](../../../components/data_processing/autorag/text_extraction/README.md)
4. [Documents indexing](../../../components/data_processing/autorag/documents_indexing/README.md)

## Compiling the pipeline

From the repo root:

```bash
python pipelines/data_processing/autorag/documents_indexing_pipeline/pipeline.py
```

This produces `documents_indexing_pipeline.yaml` in the same directory.
