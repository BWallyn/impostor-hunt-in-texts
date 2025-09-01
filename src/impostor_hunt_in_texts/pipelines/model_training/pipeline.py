"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline

from impostor_hunt_in_texts.pipelines.model_training.mlflow import (
    create_mlflow_experiment,
)
from impostor_hunt_in_texts.pipelines.model_training.nodes import (
    initialize_model_params,
    split_data_labels,
    train_final_model,
    train_model_bayesian_opti_cross_val,
    validate_params,
)
from impostor_hunt_in_texts.utils.utils import drop_columns


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline."""
    return Pipeline(
        nodes=[
            Node(
                func=validate_params,
                inputs={
                    "experiment_folder_path": "params:experiment_folder_path",
                    "experiment_name": "params:experiment_name",
                    "experiment_id_saved": "params:experiment_id_saved",
                    "model_name": "params:model_name",
                    "model_pca_n_components": "params:pca_n_components",
                    "n_trials": "params:n_trials",
                    "default_hyperparameters": "params:default_hyperparameters",
                    "search_space": "params:search_space",
                    "label_column": "params:label_column",
                },
                outputs=None,
                name="Validate_input_parameters_model_training",
            ),
            Node(
                func=create_mlflow_experiment,
                inputs={
                    "experiment_folder_path": "params:experiment_folder_path",
                    "experiment_name": "params:experiment_name",
                    "experiment_id": "params:experiment_id_saved",
                },
                outputs="experiment_id",
                name="Create_or_load_mlflow_experiment",
            ),
            Node(
                func=initialize_model_params,
                inputs={
                    "model_name": "params:model_name",
                    "pca_n_components": "params:pca_n_components",
                    "search_space": "params:search_space",
                    "n_trials": "params:n_trials",
                    "default_hyperparameters": "params:default_hyperparameters",
                },
                outputs="model_params",
                name="Initialize_model_parameters",
            ),
            Node(
                func=drop_columns,
                inputs={
                    "df": "df_train_features",
                    "cols_to_drop": "params:id_to_drop",
                },
                outputs="df_train_id_droped",
                name="Drop_id_columns_from_training_data",
            ),
            Node(
                func=split_data_labels,
                inputs={
                    "df": "df_train_id_droped",
                    "label_column": "params:label_column",
                },
                outputs=["x_training", "y_training"],
                name="Split_data_labels_training",
            ),
            Node(
                func=train_model_bayesian_opti_cross_val,
                inputs={
                    "x_training": "x_training",
                    "y_training": "y_training",
                    "model_params": "model_params",
                    "experiment_id": "experiment_id",
                },
                outputs="best_params",
                name="Train_model_using_cross_validation",
            ),
            Node(
                func=train_final_model,
                inputs={
                    "x_training": "x_training",
                    "y_training": "y_training",
                    "model_params": "model_params",
                    "best_params": "best_params",
                    "experiment_id": "experiment_id",
                },
                outputs=["model_id", "run_id"],
                name="Train_final_model",
            ),
        ],
        namespace="model_training",
        inputs="df_train_features",
        outputs=["model_id", "run_id"],
    )
