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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

# ===================
# ==== FUNCTIONS ====
# ===================

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
    model = HistGradientBoostingClassifier(
        **params
    )
    model.fit(df, labels)
    return model


def train_model_cross_validate(
    x_training: pd.DataFrame,
    y_training: pd.Series,
    params: dict[str, Any],
):
    """
    Train the model to predict the impostor label using cross-validation.

    Args:
        x_training (pd.DataFrame): The training features.
        y_training (pd.Series): The target labels.
        params (dict[str, Any]): The parameters for the model.
    """
    # Define the folds using stratify to maintain class distribution
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Define the metrics to be used for evaluation
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    # Set the MLflow run
    with mlflow.start_run():
        # Loop over the folds
        for train_idx, valid_idx in kf.split(x_training, y_training):
            x_train, x_valid = x_training.iloc[train_idx], x_training.iloc[valid_idx]
            y_train, y_valid = y_training[train_idx], y_training[valid_idx]

            # Train the model
            model = _train_model_hgbm(x_train, y_train, params)
            # y_pred_train = model.predict(x_train)
            y_pred_valid = model.predict(x_valid)

            # Analyze the model performance
            accuracies.append(accuracy_score(y_true=y_valid, y_pred=y_pred_valid))
            precisions.append(precision_score(y_true=y_valid, y_pred=y_pred_valid))
            recalls.append(recall_score(y_true=y_valid, y_pred=y_pred_valid))
            f1_scores.append(f1_score(y_true=y_valid, y_pred=y_pred_valid))

        # Average metrics across folds
        metrics = {
            "accuracy": np.mean(accuracies),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1_score": np.mean(f1_scores),
        }

        # Log the metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log the model
        mlflow.sklearn.log_model(model, "model")
