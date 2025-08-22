"""This is a boilerplate pipeline 'prepare_data' generated using Kedro 0.19.14"""

from kedro.pipeline import Node, Pipeline

from impostor_hunt_in_texts.pipelines.prepare_data.nodes import (
    create_dataset_test,
    create_dataset_train,
    create_datasets_dict,
    validate_input_params,
)
from impostor_hunt_in_texts.utils.utils import save_hf_datasetdict


def create_pipeline(**kwargs) -> Pipeline:
    """Create the prepare data pipeline."""
    return Pipeline(
        nodes=[
            Node(
                func=validate_input_params,
                inputs={
                    "path_data_train": "params:path_data_train",
                    "path_data_test": "params:path_data_test",
                    "path_dataset_dict": "params:path_dataset_dict",
                },
                outputs=None,
                name="Validate_input_parameters_prepare_data",
            ),
            Node(
                func=create_dataset_train,
                inputs=["df_train", "params:path_data_train"],
                outputs="dataset_train",
                name="Create_dataset_train",
            ),
            Node(
                func=create_dataset_test,
                inputs="params:path_data_test",
                outputs="dataset_test",
                name="Create_dataset_test",
            ),
            Node(
                func=create_datasets_dict,
                inputs=["dataset_train", "dataset_test"],
                outputs="dataset_dict",
                name="Create_datasets_dict",
            ),
            Node(
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
