from kfp import dsl
from kfp_components.components.data_processing.automl.tabular_data_loader import automl_data_loader
from kfp_components.components.data_processing.automl.tabular_train_test_split import tabular_train_test_split
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import leaderboard_evaluation
from kfp_components.components.training.automl.autogluon_models_full_refit import autogluon_models_full_refit
from kfp_components.components.training.automl.autogluon_models_selection import models_selection


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
