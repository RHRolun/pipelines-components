from kfp import dsl
from langchain_core.documents import Document


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=[
        "langchain-text-splitters",
        "ai4rag@git+https://github.com/IBM/ai4rag.git"
    ],
)
def documents_indexing(
    embedding_params: dict,
    embedding_model_id: str,
    collection_name: str,
    extracted_text: dsl.Input[dsl.Artifact],
    provider_id: str,
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0
):
    import os
    from pathlib import Path

    from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams
    from ai4rag.rag.vector_store.llama_stack import LSVectorStore
    from ai4rag.rag.chunking import LangChainChunker
    from llama_stack_client import LlamaStackClient

    params = LSEmbeddingParams(**embedding_params)

    client = LlamaStackClient(
        base_url=os.getenv("LLAMA_STACK_CLIENT_BASE_URL"),
        api_key=os.getenv("LLAMA_STACK_CLIENT_API_KEY"),
    )

    paths = list(Path(extracted_text.path).glob("*.md"))
    documents = [
        Document(
            page_content=p.read_text(encoding="utf-8", errors="replace"),
            metadata={"document_id": p.stem},
        )
        for p in paths
    ]

    chunker = LangChainChunker(method=chunking_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split_documents(documents)

    embedding_model = LSEmbeddingModel(client=client, model_id=embedding_model_id, params=params)
    ls_vectorstore = LSVectorStore(
        embedding_model=embedding_model,
        client=client,
        provider_id=provider_id,
        distance_metric=distance_metric,
        reuse_collection_name=collection_name
    )
    ls_vectorstore.add_documents(chunks)

