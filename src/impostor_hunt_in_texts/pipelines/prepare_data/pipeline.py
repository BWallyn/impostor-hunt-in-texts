"""
This is a boilerplate pipeline 'prepare_data'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from impostor_hunt_in_texts.pipelines.prepare_data.nodes import (
    create_dataset_test,
    create_dataset_train,
)


def create_pipeline(**kwargs) -> Pipeline:
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
        ],
        inputs=["df_train"],
        namespace="prepare_data",
    )
