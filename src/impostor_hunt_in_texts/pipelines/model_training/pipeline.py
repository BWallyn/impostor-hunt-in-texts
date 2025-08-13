"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline

from impostor_hunt_in_texts.pipelines.model_training.nodes import (
    initialize_model_params,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline."""
    return Pipeline(
        nodes=[
            Node(
                func=initialize_model_params,
                inputs={
                    "model_name": "params:model_name",
                    "params": "params:model_params",
                },
                outputs="model_params",
                name="Initialize_model_parameters",
            )
        ],
        namespace="model_training",
    )
