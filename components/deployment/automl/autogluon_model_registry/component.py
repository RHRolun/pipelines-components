from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
    packages_to_install=[
        "autogluon.tabular==1.5.0",
        "catboost==1.2.8",
        "fastai==2.8.5",
        "lightgbm==4.6.0",
        "torch==2.9.1",
        "xgboost==3.1.3",
        "model-registry",
    ],
)
def autogluon_model_registry(
    best_model: str,
    models: List[dsl.Model],
    username: str,
    cluster_domain: str,
    version: str,
    registered_model_name: str,
    deployment_artifact: dsl.Output[dsl.Model],
) -> NamedTuple("outputs", model_uri=str):
    """Clone the best AutoGluon model for deployment, push it as a modelcar, and register it.

    Finds the best model artifact by name, clones it for deployment using AutoGluon's
    clone_for_deployment() (producing a minimal inference-only predictor), then pushes
    it as an OCI modelcar and registers it in the OpenDataHub Model Registry.

    OCI registry authentication is handled via the REGISTRY_AUTH_FILE environment variable,
    which should point to a Docker-format auth JSON file (e.g. produced by `oc registry login`).

    Args:
        best_model: Display name of the best model (e.g. "LightGBM_BAG_L1_FULL"), as returned by leaderboard_evaluation.
        models: List of Model artifacts from the parallel refit step.
        username: OpenShift username; used to construct the model registry URL, OCI reference, and as the registry author.
        cluster_domain: OpenShift cluster domain (e.g. "apps.cluster.example.com"); used to construct service URLs.
        version: Version string for the registered model (e.g. "1.0.0").
        registered_model_name: Name under which the model will be registered in the Model Registry.
        deployment_artifact: Output artifact containing the deployment-cloned predictor files.

    Returns:
        model_uri: OCI image reference of the pushed modelcar.

    Raises:
        ValueError: If best_model is not found in the provided model artifacts.

    Example:
        from kfp import dsl
        from components.deployment.automl.autogluon_model_registry import autogluon_model_registry

        @dsl.pipeline(name="register-pipeline")
        def register_pipeline(models, best_model):
            autogluon_model_registry(
                best_model=best_model,
                models=models,
                username="jdoe",
                cluster_domain="apps.cluster.example.com",
                version="1.0.0",
                registered_model_name="loan-default-predictor",
            )
    """  # noqa: E501
    from pathlib import Path

    from autogluon.tabular import TabularPredictor
    from model_registry import ModelRegistry
    from model_registry.utils import OCIParams

    model_registry_url = f"https://{username}-registry-rest.{cluster_domain}"
    oci_image_ref = f"default-route-openshift-image-registry.{cluster_domain}/{username}-canopy/{registered_model_name}:{version}"  # noqa: E501

    # --- Find the best model artifact ---
    best_model_artifact = next(
        (m for m in models if m.metadata.get("display_name") == best_model),
        None,
    )
    if best_model_artifact is None:
        available = [m.metadata.get("display_name") for m in models]
        raise ValueError(f"Best model '{best_model}' not found in artifacts. Available: {available}")

    # --- Load predictor and clone for deployment ---
    predictor_path = Path(best_model_artifact.path) / best_model / "predictor"
    predictor = TabularPredictor.load(str(predictor_path))

    deploy_path = Path(deployment_artifact.path) / "deployment_predictor"
    predictor.clone_for_deployment(path=str(deploy_path))

    # --- Build metadata from artifact context ---
    context = best_model_artifact.metadata.get("context", {})
    task_type = context.get("task_type", "unknown")
    label_column = context.get("label_column", "unknown")
    metrics = context.get("metrics", {}).get("test_data", {})

    # --- Push modelcar and register ---
    mr = ModelRegistry(model_registry_url, author=username)
    mr.upload_artifact_and_register_model(
        name=registered_model_name,
        model_files_path=str(deploy_path),
        author=username,
        version=version,
        description=f"AutoGluon tabular model '{best_model}' trained for {task_type} on column '{label_column}'.",
        upload_params=OCIParams(base_image="busybox", oci_ref=oci_image_ref),
        model_format_name="autogluon",
        model_format_version="1",
        metadata={
            "source_model": best_model,
            "task_type": task_type,
            "label_column": label_column,
            **{f"metric_{k}": str(v) for k, v in metrics.items()},
        },
    )

    deployment_artifact.metadata["display_name"] = registered_model_name
    deployment_artifact.metadata["version"] = version
    deployment_artifact.metadata["oci_ref"] = oci_image_ref

    return NamedTuple("outputs", model_uri=str)(model_uri=oci_image_ref)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_model_registry,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
