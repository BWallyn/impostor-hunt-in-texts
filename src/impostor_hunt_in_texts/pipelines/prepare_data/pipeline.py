"""This is a boilerplate pipeline 'prepare_data' generated using Kedro 0.19.14"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from impostor_hunt_in_texts.pipelines.prepare_data.nodes import (
    create_dataset_test,
    create_dataset_train,
    create_datasets_dict
)
from impostor_hunt_in_texts.utils.utils import save_hf_datasetdict


def create_pipeline(**kwargs) -> Pipeline:
    """Create the prepare data pipeline."""
    return pipeline(
        pipe=[
            node(
                func=create_dataset_train,
                inputs=["df_train", "params:path_data_train"],
                outputs="dataset_train",
                name="Create_dataset_train",
            ),
            node(
                func=create_dataset_test,
                inputs="params:path_data_test",
                outputs="dataset_test",
                name="Create_dataset_test",
            ),
            node(
                func=create_datasets_dict,
                inputs=["dataset_train", "dataset_test"],
                outputs="dataset_dict",
                name="Create_datasets_dict",
            ),
            node(
                func=save_hf_datasetdict,
                inputs=["dataset_dict", "params:path_dataset_dict"],
                outputs="dict_metadata_datasets",
                name="Save_dataset_dict",
            ),
        ],
        inputs=["df_train"],
        outputs="dict_metadata_datasets",
        namespace="prepare_data",
    )
