"""Input validation component for the documents RAG optimization pipeline."""

from typing import List, Optional

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
)
def input_validation(
    test_data_secret_name: str,
    test_data_bucket_name: str,
    test_data_key: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    input_data_key: str,
    llama_stack_secret_name: str,
    optimization_metric: str,
    optimization_max_rag_patterns: int,
    embeddings_models: Optional[List[str]],
    generation_models: Optional[List[str]],
    llama_stack_vector_database_id: Optional[str],
    chat_model_url: Optional[str],
    chat_model_token: Optional[str],
    embedding_model_url: Optional[str],
    embedding_model_token: Optional[str],
) -> None:
    """Validate parameters for the documents RAG optimization pipeline.

    This component is intended as the first step of the documents RAG optimization
    pipeline. It checks that all required string parameters are non-empty,
    optimization_metric is one of the supported values, optimization_max_rag_patterns
    is within the allowed range, and optional lists (if provided) contain non-empty
    strings.

    Args:
        test_data_secret_name: Name of the Kubernetes secret holding S3-compatible credentials for
            test data access. The following environment variables are required:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION.
        test_data_bucket_name: S3 (or compatible) bucket name for the test data file.
        test_data_key: Object key (path) of the test data JSON file in the test data bucket.
        input_data_secret_name: Name of the Kubernetes secret holding S3-compatible credentials
            for input document data access. The following environment variables are required:
            AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION.
        input_data_bucket_name: S3 (or compatible) bucket name for the input documents.
        input_data_key: Object key (path) of the input documents in the input data bucket.
        llama_stack_secret_name: Name of the Kubernetes secret for llama-stack API connection.
        embeddings_models: Optional list of embedding model identifiers to use in the search space.
        generation_models: Optional list of foundation/generation model identifiers to use in the
            search space.
        optimization_metric: Quality metric used to optimize RAG patterns. Supported values:
            "faithfulness", "answer_correctness", "context_correctness".
        optimization_max_rag_patterns: Maximum number of RAG patterns to generate. Passed to ai4rag
            (max_number_of_rag_patterns). Defaults to 8.
        llama_stack_vector_database_id: Optional vector database id (e.g., registered in llama-stack Milvus).
            If not provided, an in-memory database may be used.
        chat_model_url: Inference endpoint URL for the chat/generation model (OpenAI-compatible endpoint).
        chat_model_token: API token or key for authenticating with the chat model endpoint.
        embedding_model_url: Inference endpoint URL for the embedding model.
        embedding_model_token: API token or key for authenticating with the embedding model endpoint.

    Raises:
        ValueError: If any parameter fails validation.
    """
    import sys
    import logging

    logger = logging.getLogger("Input Validation component")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})
    MIN_OPTIMIZATION_MAX_RAG_PATTERNS = 1
    MAX_OPTIMIZATION_MAX_RAG_PATTERNS = 20

    errors: List[str] = []

    def require_non_empty(**fields):
        for name, value in fields.items():
            if not value:
                errors.append(f"{name} must be a non-empty string.")

    if llama_stack_secret_name is not None:
        # Llama Stack scenario
        require_non_empty(
            test_data_secret_name=test_data_secret_name,
            test_data_bucket_name=test_data_bucket_name,
            test_data_key=test_data_key,
            input_data_secret_name=input_data_secret_name,
            input_data_bucket_name=input_data_bucket_name,
            input_data_key=input_data_key,
            llama_stack_secret_name=llama_stack_secret_name,
        )

        if embeddings_models is not None:
            if not isinstance(embeddings_models, list):
                errors.append("embeddings_models must be a list.")
            else:
                for i, m in enumerate(embeddings_models):
                    if not m:
                        errors.append(f"embeddings_models[{i}] must be a non-empty string.")

        if generation_models is not None:
            if not isinstance(generation_models, list):
                errors.append("generation_models must be a list when provided.")
            else:
                for i, m in enumerate(generation_models):
                    if not m:
                        errors.append(f"generation_models[{i}] must be a non-empty string.")

    else:
        # Chroma scenario
        require_non_empty(
            test_data_secret_name=test_data_secret_name,
            test_data_bucket_name=test_data_bucket_name,
            test_data_key=test_data_key,
            input_data_secret_name=input_data_secret_name,
            input_data_bucket_name=input_data_bucket_name,
            input_data_key=input_data_key,
            chat_model_url=chat_model_url,
            chat_model_token=chat_model_token,
            embedding_model_url=embedding_model_url,
            embedding_model_token=embedding_model_token,
        )

    if llama_stack_vector_database_id is not None and not str(llama_stack_vector_database_id).strip():
        errors.append("llama_stack_vector_database_id must be non-empty when provided.")

    if not test_data_key.strip().endswith(".json"):
        errors.append("test_data_key must point to a JSON file (path must end with .json).")

    if optimization_metric not in SUPPORTED_OPTIMIZATION_METRICS:
        errors.append(
            f"optimization_metric must be one of {sorted(SUPPORTED_OPTIMIZATION_METRICS)}; got {optimization_metric!r}."
        )

    try:
        max_rag = int(optimization_max_rag_patterns)
    except (TypeError, ValueError):
        errors.append(
            f"optimization_max_rag_patterns must be an integer; got {optimization_max_rag_patterns!r}."
        )
    else:
        if not (MIN_OPTIMIZATION_MAX_RAG_PATTERNS <= max_rag <= MAX_OPTIMIZATION_MAX_RAG_PATTERNS):
            errors.append(
                f"optimization_max_rag_patterns must be between {MIN_OPTIMIZATION_MAX_RAG_PATTERNS} and "
                f"{MAX_OPTIMIZATION_MAX_RAG_PATTERNS}; got {max_rag}."
            )

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(errors))

if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        input_validation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
