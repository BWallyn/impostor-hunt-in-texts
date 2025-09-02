"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline

from impostor_hunt_in_texts.pipelines.model_prediction.nodes import (
    create_predictions_df,
)
from impostor_hunt_in_texts.utils.utils import drop_columns
from impostor_hunt_in_texts.utils.utils_model import load_model


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model prediction pipeline."""
    return Pipeline(
        nodes=[
            Node(
                func=load_model,
                inputs="model_id",
                outputs="model",
                name="Load_trained_model",
            ),
            Node(
                func=drop_columns,
                inputs={
                    "df": "df_test_features",
                    "cols_to_drop": "params:id_to_drop",
                },
                outputs="df_test_id_droped",
                name="Drop_id_columns_from_test_data",
            ),
            Node(
                func=create_predictions_df,
                inputs={
                    "model": "model",
                    "df": "df_test_features",
                },
                outputs="df_pred_test",
                name="Create_predictions_test",
            ),
        ],
        namespace="model_prediction",
        inputs=["model_id", "df_test_features"],
        outputs="df_pred_test",
    )
