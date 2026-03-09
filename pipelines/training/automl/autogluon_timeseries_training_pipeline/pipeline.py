from kfp import dsl


@dsl.pipeline(
    name="autogluon-timeseries-training-pipeline",
    description=(
        "Timeseries Pipeline" #TODO add a description
    ),
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size="1Gi",  # TODO: change to recommended size
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    "storageClassName": "gp3-csi",  # or 'gp3', 'fast', etc.
                    "accessModes": ["ReadWriteOnce"],
                }
            ),
        ),
    ),
)
def autogluon_timeseries_training_pipeline(
    taget: str
):
    """
    #TODO  add documentation
    """  # noqa: E501
    pass


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_training_pipeline,
        package_path=__file__.replace(".py", "_pipeline.yaml"),
    )
