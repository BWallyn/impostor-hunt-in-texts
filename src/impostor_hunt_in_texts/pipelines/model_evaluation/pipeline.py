"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline

from impostor_hunt_in_texts.pipelines.model_evaluation.nodes import (
    get_feature_importance,
    log_tables_to_mlflow,
)
from impostor_hunt_in_texts.utils.utils_model import load_model_sklearn


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model evaluation pipeline."""
    return Pipeline(
        nodes=[
            Node(
                func=load_model_sklearn,
                inputs="model_id",
                outputs="model",
                name="Load_trained_model",
            ),
            Node(
                func=get_feature_importance,
                inputs={
                    "model": "model",
                    "df": "df_test_features",
                },
                outputs="dict_tables_to_log",
                name="Get_feature_importance",
            ),
            Node(
                func=log_tables_to_mlflow,
                inputs={
                    "dict_tables": "dict_tables_to_log",
                    "run_id": "run_id",
                },
                outputs=None,
                name="Log_tables_to_MLflow",
            )
        ],
        namespace="model_evaluation",
        inputs=["model_id", "df_test_features", "run_id"],
    )
