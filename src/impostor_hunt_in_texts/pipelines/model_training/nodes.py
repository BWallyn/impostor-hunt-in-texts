"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""
# =================
# ==== IMPORTS ====
# =================

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from impostor_hunt_in_texts.pipelines.model_training.model_params import ModelParams
from impostor_hunt_in_texts.pipelines.model_training.validate_params import (
    ValidateParams,
)

# ===================
# ==== FUNCTIONS ====
# ===================

def validate_params(  # noqa: PLR0913
    experiment_folder_path: str,
    experiment_name: str,
    experiment_id_saved: str,
    model_name: str,
    model_params: dict[str, Any],
    label_column: str,
) -> None:
    """Validate the input parameters for the model training pipeline."""
    ValidateParams(
        experiment_folder_path=experiment_folder_path,
        experiment_name=experiment_name,
        experiment_id_saved=experiment_id_saved,
        model_name=model_name,
        model_params=model_params,
        label_column=label_column,
    )


def initialize_model_params(model_name: str, params: dict[str, Any]) -> ModelParams:
    """
    Initialize the model parameters.

    Args:
        model_name (str): The name of the model to use.
        params (dict[str, Any]): The parameters for the model.

    Returns:
        (ModelParams): The initialized model parameters.
    """
    return ModelParams(
        model_name=model_name,
        params=params,
    )


def _train_model_rf(df: pd.DataFrame, labels: pd.Series, params: dict[str, Any]) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model.

    Args:
        df (pd.DataFrame): The training data.
        labels (pd.Series): The target labels.
        params (dict[str, Any]): The parameters for the model.

    Returns:
        (RandomForestClassifier): The trained model.
    """
    # Initialize the model
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(df, labels)
    return model


def _train_model_hgbm(df: pd.DataFrame, labels: pd.Series, params: dict[str, Any]) -> HistGradientBoostingClassifier:
    """
    Train a HistGradientBoostingClassifier model.

    Args:
        df (pd.DataFrame): The training data.
        labels (pd.Series): The target labels.
        params (dict[str, Any]): The parameters for the model.

    Returns:
        (HistGradientBoostingClassifier): The trained model.
    """
    # Initialize the model
    model = HistGradientBoostingClassifier(**params, random_state=42)
    model.fit(df, labels)
    return model


def _create_pipeline_model(model_name: str, n_comp: int, params: dict[str, Any]) -> Pipeline:
    """
    Create scikit-learn pipeline with PCA and a classifier.

    Args:
        model_name (str): Name of the model to use as a classifier.
        n_comp (int): Number of components to get from the PCA.
        params (dict[str, Any]): Parameters for the model.

    Returns:
        (Pipeline): The scikit learn pipeline containing a PCA and a classifier.
    """
    estimators = [
        ("reduce_dim", PCA(n_components=n_comp)),
    ]
    if model_name == "HistGradientBoostingClassifier":
        estimators.append(
            ("classifier", HistGradientBoostingClassifier(**params, random_state=42))
        )
    elif model_name == "RandomForestClassifier":
        estimators.append(
            ("classifier", RandomForestClassifier(**params, random_state=42))
        )
    return Pipeline(estimators)


def split_data_labels(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the DataFrame into features and labels.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        label_column (str): The name of the column containing the labels.

    Returns:
        (tuple[pd.DataFrame, pd.Series]): A tuple containing the features DataFrame and the labels Series.
    """
    return df.drop(columns=[label_column]), df[label_column]


def train_model_cross_validate(
    x_training: pd.DataFrame,
    y_training: pd.Series,
    model_params: ModelParams,
    experiment_id: str,
):
    """
    Train the model to predict the impostor label using cross-validation.

    Args:
        x_training (pd.DataFrame): The training features.
        y_training (pd.Series): The target labels.
        model_params (ModelParams): The parameters for the model.
        experiment_id (str): The id of the MLflow experiment.
    """
    # Define the folds using stratify to maintain class distribution
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Define the metrics to be used for evaluation
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Set the MLflow run
    with mlflow.start_run(experiment_id=experiment_id):
        # Loop over the folds
        for train_idx, valid_idx in kf.split(x_training, y_training):
            x_train, x_valid = x_training.iloc[train_idx], x_training.iloc[valid_idx]
            y_train, y_valid = y_training[train_idx], y_training[valid_idx]

            # Train the model
            if model_params.model_name == "RandomForestClassifier":
                model = _train_model_rf(x_train, y_train, model_params.params)
            elif model_params.model_name == "HistGradientBoosting":
                model = _train_model_hgbm(x_train, y_train, model_params.params)
            # y_pred_train = model.predict(x_train)
            y_pred_valid = model.predict(x_valid)

            # Analyze the model performance
            accuracies.append(accuracy_score(y_true=y_valid, y_pred=y_pred_valid))
            precisions.append(precision_score(y_true=y_valid, y_pred=y_pred_valid))
            recalls.append(recall_score(y_true=y_valid, y_pred=y_pred_valid))
            f1_scores.append(f1_score(y_true=y_valid, y_pred=y_pred_valid))

        # Average metrics across folds
        metrics = {
            "accuracy_valid": np.mean(accuracies),
            "precision_valid": np.mean(precisions),
            "recall_valid": np.mean(recalls),
            "f1_score_valid": np.mean(f1_scores),
        }

        # Log the metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log the model
        mlflow.sklearn.log_model(model, name="model", input_example=x_train.sample(5))


def train_final_model(
    x_training: pd.DataFrame,
    y_training: pd.Series,
    model_params: ModelParams,
    experiment_id: str,
) -> None:
    """
    Train the final model on the entire training set and evaluate it on the test set.

    Args:
        x_training (pd.DataFrame): The training features.
        y_training (pd.Series): The target labels for training.
        model_params (ModelParams): The parameters for the model.
        experiment_id (str): The id of the MLflow experiment.

    Returns:
        None
    """
    # Set the MLflow run
    tags = {"model_name": "final_model", "model_type": model_params.model_name}
    with mlflow.start_run(experiment_id=experiment_id, tags=tags):
        # Train the model
        if model_params.model_name == "RandomForestClassifier":
            model = _train_model_rf(x_training, y_training, model_params.params)
        elif model_params.model_name == "HistGradientBoosting":
            model = _train_model_hgbm(x_training, y_training, model_params.params)
        y_pred_training = model.predict(x_training)

        # Compute metrics
        metrics = {
            "accuracy_training": accuracy_score(y_true=y_training, y_pred=y_pred_training),
            "precision_training": precision_score(y_true=y_training, y_pred=y_pred_training),
            "recall_training": recall_score(y_true=y_training, y_pred=y_pred_training),
            "f1_score_training": f1_score(y_true=y_training, y_pred=y_pred_training),
        }

        # Log the metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log the model
        mlflow.sklearn.log_model(model, name="model", input_example=x_training.sample(5))
