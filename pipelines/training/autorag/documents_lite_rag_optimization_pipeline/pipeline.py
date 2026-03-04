from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.autorag.documents_discovery import documents_discovery
from kfp_components.components.data_processing.autorag.test_data_loader import test_data_loader
from kfp_components.components.data_processing.autorag.text_extraction import text_extraction
from kfp_components.components.training.autorag.leaderboard_evaluation import leaderboard_evaluation
from kfp_components.components.training.autorag.rag_templates_optimization.component import rag_templates_optimization
from kfp_components.components.training.autorag.search_space_preparation.component import search_space_preparation

SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})


@dsl.pipeline(
    name="documents-lite-rag-optimization-pipeline",
    description=(
        "Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications "
        "(lite version). The lite version does not use llama-stack API for inference and vector database "
        "operations."
    ),
)
def documents_lite_rag_optimization_pipeline(
    test_data_secret_name: str = "autorag-minio-secret",
    test_data_bucket_name: str = "autorag-input-data",
    test_data_key: str = "test_data_small/benchmark.json",
    input_data_secret_name: str = "autorag-minio-secret",
    input_data_bucket_name: str = "autorag-input-data",
    input_data_key: str = "test_data_small",
    chat_model_url: str = "https://redhataillama-31-8b-instruct-ai-eng-cracow.apps.rosa.ai-eng-gpu.socc.p3.openshiftapps.com",
    chat_model_token: str = "eyJhbGciOiJSUzI1NiIsImtpZCI6IlQ5RXNsTDhhN3UyLVlEMW1Rb21ESXB0UXdVRmpEOTRDVjZmR0VMZjZUdWsifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJhaS1lbmctY3JhY293Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImxsYW1hLXRva2VuLXJlZGhhdGFpbGxhbWEtMzEtOGItaW5zdHJ1Y3Qtc2EiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoicmVkaGF0YWlsbGFtYS0zMS04Yi1pbnN0cnVjdC1zYSIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjUyMzM0ZjgxLTE4YzctNGI0Yy1iZTExLWUzMmVmMTBkY2U2NCIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDphaS1lbmctY3JhY293OnJlZGhhdGFpbGxhbWEtMzEtOGItaW5zdHJ1Y3Qtc2EifQ.mm76D2l-DCBlpjzOPHpN6-PBDXtpwS4g9ThVk_m0Wdfuxlkr6-ytQA3SB-3fD2Vuy3gbdfOmcZxriZYK6O-ztJRJEEZBLPD5n2jwIudRtAjme3TnF6GgNZSP075r1Rd2RR4fgQmMk1LM0_iToREhTi3bAlMewBDknE3Ns4NBJI_j5iJuaPjluesl2qVHN6jdjKQEo-8KT1LfqKxPVH8KJOKP4vaLuzauCPmxGcYW3vNEsZRDwfdTXq1Yx-6wY5S-iQ81xol4YW6BV-v9D8D_hkpo1pp4NJT5EUu_i0lSGjlCYMDdzXtR6GONX-E0806oqaPOqGl6IRAJmiaw-2hKNTb6300YqGZb7N2Q9APNTKB-1EeCVi4X350l9IKNDG5xljDP5EC9AR8y39K4o-h6WFs4r6fF5B8YrsvblAdhjcJ6PIwJBrUei0VWnbyf8fYZd8-c1LkE5Qka7FK6MzJ6DN0P3NFIFcoTbN6JdRl5tdzhPfTHaSVe8qSk7tAi0BTlS_8W0aPYXzgSmKez2tsVfSKNAVeJof40xpuhf4oBo8Pazt7hf7pXjI0tKxy8LyXliXgEc1X2oEZqHgdcjuyCQc-nhBbKYmQH88WZAiSuTynh5gFlKBDOGHe_KmtPHFWVwdQCTJGZZ6eJmpF2e5S2oU_nWhoNSsez3E05AXOEyNU",
    embedding_model_url: str = "https://granite-278m-multilingual-1-ai-eng-cracow.apps.rosa.ai-eng-gpu.socc.p3.openshiftapps.com",
    embedding_model_token: str = "eyJhbGciOiJSUzI1NiIsImtpZCI6IlQ5RXNsTDhhN3UyLVlEMW1Rb21ESXB0UXdVRmpEOTRDVjZmR0VMZjZUdWsifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJhaS1lbmctY3JhY293Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6ImVtYmVkZGluZy10b2tlbi1ncmFuaXRlLTI3OG0tbXVsdGlsaW5ndWFsLTEtc2EiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiZ3Jhbml0ZS0yNzhtLW11bHRpbGluZ3VhbC0xLXNhIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiNGRjOWFlMmUtMzE2YS00MjMzLWI2MGYtYjg4NzYyNGY4YWI3Iiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OmFpLWVuZy1jcmFjb3c6Z3Jhbml0ZS0yNzhtLW11bHRpbGluZ3VhbC0xLXNhIn0.wxjmP7es9xGtAWQlQY_zx3Qqlp4PcpwI5xwjPl4inb3kldkGnR5OLVt4wtdDcNy0nWloU5wZnZfobz5U1xc5frtbqqAdaij1HZggUFTXgmgLOktBYqp7sWqOBnisiWu7mpxIX1UU7XOzV4V2DLMRHiaZELRx6oUq8NMuhYxKIFwtkhq9jb-SOfM-no1EhLxih1aFe-hiM-8EgeMxHRcklerfoAxvgr11RP38o5Ro0c0drSdLTgbVGLC32OGe0JpJjdEQUqEx8b85Qtm4DMtERW77BLbWVAjschoGQ6by3budsIb4RAyPe24tRDRjmRoGnntFo5uZFAGTHyj_XQVl6f-dkI5WMAk-T9iMRht1GLduToZFcmnl5JPLEV_ZUAdXc0VKsmBKSzJsKTD1i7Pcfed2xK_XnWyBQbRzW8jsXlsK-PuM8oUdlL7E_A3JcPDGzxPtgTh4rDsHutizT27KbW3JlDCq13O38sbV6HMA4MpmuCOLEGsEOTO_n5lVI2a4D3tdZVbGXjEWA2eUz1N-5to8n61Gmo8TH3rCzFBN-lPHe0VP3cSQfadrnuzYBD0br3Tm9zMnhgl4sQGipyccBp-53YVDR7f6i66Ntq-1IZDihOUKeXH_GI_q91GAOIZU03Z7eMiCwMWXlT4j_pOkIdzkEgZtakUj2UjhNGkSM-4",
    optimization_metric: str = "faithfulness",
):
    """Automated system for building and optimizing Retrieval-Augmented Generation (RAG) applications.

    The Documents Lite RAG Optimization Pipeline is an automated system for building and optimizing
    Retrieval-Augmented Generation (RAG) applications within Red Hat OpenShift AI. It leverages
    Kubeflow Pipelines to orchestrate the optimization workflow, using the ai4rag optimization
    engine to systematically explore RAG configurations and identify the best performing parameter
    settings based on an upfront-specified quality metric.

    The system integrates with OpenAI API for inference and in-memory ChromaDB vector database operations,
    producing optimized RAG patterns as artifacts that can be deployed and used for production
    RAG applications.

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
        chat_model_url: Inference endpoint URL for the chat/generation model (OpenAI-compatible endpoint).
        chat_model_token: API token or key for authenticating with the chat model endpoint.
        embedding_model_url: Inference endpoint URL for the embedding model.
        embedding_model_token: API token or key for authenticating with the embedding model endpoint.
        optimization_metric: Quality metric used to optimize RAG patterns. Supported values:
            "faithfulness", "answer_correctness", "context_correctness". Defaults to "faithfulness".
        embeddings_models: Optional list of embedding model IDs for the search space. If not set,
            defaults to a single model so the pipeline runs without manual model discovery.
        generation_models: Optional list of foundation/generation model IDs for the search space.
            If not set, defaults to a single model so the pipeline runs without manual model discovery.
    """
    test_data_loader_task = test_data_loader(
        test_data_bucket_name=test_data_bucket_name,
        test_data_path=test_data_key,
    )

    documents_discovery_task = documents_discovery(
        input_data_bucket_name=input_data_bucket_name,
        input_data_path=input_data_key,
        test_data=test_data_loader_task.outputs["test_data"],
    )

    text_extraction_task = text_extraction(
        documents_descriptor=documents_discovery_task.outputs["discovered_documents"],
    )

    for task, secret_name in zip(
        [test_data_loader_task, documents_discovery_task, text_extraction_task],
        [test_data_secret_name, input_data_secret_name, input_data_secret_name],
    ):
        use_secret_as_env(
            task,
            secret_name=secret_name,
            secret_key_to_env={
                "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
                "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
                "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
            },
        )

    mps_task = search_space_preparation(
        test_data=test_data_loader_task.outputs["test_data"],
        extracted_text=text_extraction_task.outputs["extracted_text"],
        chat_model_url=chat_model_url,
        chat_model_token=chat_model_token,
        embedding_model_url=embedding_model_url,
        embedding_model_token=embedding_model_token,
    )

    hpo_task = rag_templates_optimization(
        extracted_text=text_extraction_task.outputs["extracted_text"],
        test_data=test_data_loader_task.outputs["test_data"],
        search_space_prep_report=mps_task.outputs["search_space_prep_report"],
        chat_model_url=chat_model_url,
        chat_model_token=chat_model_token,
        embedding_model_url=embedding_model_url,
        embedding_model_token=embedding_model_token,
        optimization_settings={"metric": optimization_metric},
    )

    leaderboard_evaluation(
        rag_patterns=hpo_task.outputs["rag_patterns"],
        optimization_metric=optimization_metric,
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_lite_rag_optimization_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
