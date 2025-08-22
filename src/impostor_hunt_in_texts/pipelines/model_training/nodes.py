"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""
# =================
# ==== IMPORTS ====
# =================

import logging
from functools import partial
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna import Trial
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

# Options
logger = logging.getLogger(__name__)

# ===================
# ==== FUNCTIONS ====
# ===================

def validate_params(  # noqa: PLR0913
    experiment_folder_path: str,
    experiment_name: str,
    experiment_id_saved: str,
    model_name: str,
    model_pca_n_components: int,
    n_trials: int,
    search_space: dict[str, Any],
    label_column: str,
) -> None:
    """Validate the input parameters for the model training pipeline."""
    ValidateParams(
        experiment_folder_path=experiment_folder_path,
        experiment_name=experiment_name,
        experiment_id_saved=experiment_id_saved,
        model_name=model_name,
        model_pca_n_components=model_pca_n_components,
        n_trials=n_trials,
        search_space=search_space,
        label_column=label_column,
    )


def initialize_model_params(model_name: str, pca_n_components: int, search_space: dict[str, Any], n_trials: int) -> ModelParams:
    """
    Initialize the model parameters.

    Args:
        model_name (str): The name of the model to use.
        pca_n_components (int): The number of components to get from the PCA.
        search_space (dict[str, Any]): The search space of the parameters for the model.
        n_trials (int): Number of trials for the bayesian optimization.

    Returns:
        (ModelParams): The initialized model parameters.
    """
    return ModelParams(
        model_name=model_name,
        pca_n_components=pca_n_components,
        search_params=search_space,
        n_trials=n_trials,
    )


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


def _train_model_pipeline(
    pipe: Pipeline, df: pd.DataFrame, labels: pd.Series,
) -> Pipeline:
    """
    Train a scikit-learn pipeline.

    Args:
        pipe (Pipeline): The scikit-learn pipeline to train.
        df (pd.DataFrame): The training data.
        labels (pd.Series): The target labels.

    Returns:
        (Pipeline): The trained pipeline.
    """
    pipe.fit(df, labels)
    return pipe


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

            # Define the model pipeline
            pipe = _create_pipeline_model(
                model_name=model_params.model_name,
                n_comp=model_params.pca_n_components,
                params=model_params.params,
            )
            # Train the model
            pipe = _train_model_pipeline(pipe, x_train, y_train)
            # y_pred_train = model.predict(x_train)
            y_pred_valid = pipe.predict(x_valid)

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
        mlflow.sklearn.log_model(pipe, name="model", input_example=x_train.sample(5))


def _run_cross_validation(  # noqa: PLR0913
    x_training: pd.DataFrame,
    y_training: pd.Series,
    experiment_id: str,
    parent_run_id: str,
    model_name: str,
    pca_n_components: int,
    params: dict[str, Any],
) -> tuple[Pipeline, dict[str, float]]:
    """
    Run cross-validation on the training data.

    Args:
        x_training (pd.DataFrame): The training features.
        y_training (pd.Series): The target labels.
        experiment_id (str): The id of the MLflow experiment.
        parent_run_id (str): The id of the parent MLflow run.
        model_name (str): The name of the model to use.
        pca_n_components (int): The number of components to get from the PCA.
        params (dict[str, Any]): The parameters for the model.

    Returns:
        pipe (Pipeline): The trained pipeline.
        metrics (dict[str, float]): The metrics computed during cross-validation.
    """
    # Set the MLflow nested run
    logger.info(f"Train pipeline model using {model_name} with parameters {params} and with {pca_n_components} PCA components.")
    with mlflow.start_run(experiment_id=experiment_id, nested=True, parent_run_id=parent_run_id):
        # Define the folds using stratify to maintain class distribution
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Define the metrics to be used for evaluation
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        # Loop over the folds
        for train_idx, valid_idx in kf.split(x_training, y_training):
            x_train, x_valid = x_training.iloc[train_idx], x_training.iloc[valid_idx]
            y_train, y_valid = y_training[train_idx], y_training[valid_idx]

            # Define the model pipeline
            pipe = _create_pipeline_model(
                model_name=model_name,
                n_comp=pca_n_components,
                params=params,
            )
            # Train the model
            pipe = _train_model_pipeline(pipe, x_train, y_train)
            # y_pred_train = model.predict(x_train)
            y_pred_valid = pipe.predict(x_valid)

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
        mlflow.sklearn.log_model(pipe, name="model", input_example=x_train.sample(5))

    return pipe, metrics


def _build_search_space(
    trial: Trial,
    search_params: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the search space for hyperparameter optimization based on the provided sampling parameters.

    Args:
        trial (Trial): The Optuna trial object.
        search_params (dict[str, Any]): The search space for hyperparameters tuning.

    Returns:
        (dict[str, Any]): A dictionary containing the hyperparameters to be optimized.
    """
    hyperparams = {}

    for hyperparam_name, sampling_params in search_params.items():
        if sampling_params["sampling_type"] == "categorical":
            hyperparams[hyperparam_name] = eval(  # noqa: S307
                f"trial.suggest_{sampling_params['sampling_type']}('{hyperparam_name}', {sampling_params['choices']})"
            )
        else:
            hyperparams[hyperparam_name] = eval(  # noqa: S307
                f"trial.suggest_{sampling_params['sampling_type']}('{hyperparam_name}', {sampling_params['min']}, {sampling_params['max']})"
            )
    return hyperparams


def optimize_hyperparams(  # noqa: PLR0913
    trial: Trial,
    experiment_id: str,
    run_id: str,
    x_training: pd.DataFrame,
    y_training: pd.Series,
    model_params: ModelParams,
) -> float:
    """
    Optimize the hyperparameters using Bayesian Optimization with Optuna.

    This function sets the hyperparameters based on the search parameters provided,
    trains the pipeline and returns the metric on the validation set.

    Args:
        trial (Trial): Trial for bayesian optimization
        experiment_id (str): Id of the MLflow experiment
        run_id (str): Id of the MLflow run
        x_training (pd.DataFrame): The training features.
        y_training (pd.Series): The target labels.
        model_params (ModelParams): The parameters for the model creation and bayesian optimization.

    Returns:
        rmse_valid (float): Root Mean Squared Error on the validation set
    """
    optimize_params = _build_search_space(
        trial=trial,
        search_params=model_params.search_params,
    )

    # Train model mlflow
    logger.info(f"Train pipeline model using {optimize_params}")
    _, metrics = _run_cross_validation(
        x_training=x_training,
        y_training=y_training,
        experiment_id=experiment_id,
        parent_run_id=run_id,
        model_name=model_params.model_name,
        pca_n_components=model_params.pca_n_components,
        params=optimize_params,
    )
    logger.info(f"Pipeline model trained with f1-score: {metrics['f1_score_valid']}")
    # Add the best iteration as an attribute
    # trial.set_user_attr("best_iter", model.get_best_iteration())
    return metrics["f1_score_valid"]


def train_model_bayesian_opti_cross_val(
    x_training: pd.DataFrame,
    y_training: pd.Series,
    model_params: ModelParams,
    experiment_id: str,
) -> dict[str, Any]:
    """
    Train the model to predict the impostor label using bayesian optimization to find the
    best hyper parameters and cross-validation.

    Args:
        x_training (pd.DataFrame): The training features.
        y_training (pd.Series): The target labels.
        model_params (ModelParams): The parameters for the model.
        experiment_id (str): The id of the MLflow experiment.

    Returns:
        (dict[str, Any]): The best hyperparameters found during the bayesian optimization.
    """
    # Set the MLflow run
    with mlflow.start_run(experiment_id=experiment_id) as parent_run:

        # Run Bayesian optimization to find the best hyperparameters
        study = optuna.create_study(
            study_name="",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)
        )

        # Define objective function
        objective = partial(
            optimize_hyperparams,
            experiment_id=experiment_id,
            run_id=parent_run.info.run_id,
            x_training=x_training,
            y_training=y_training,
            model_params=model_params,
        )

        # Optimize
        study.optimize(objective, n_trials=model_params.n_trials, show_progress_bar=True)
        logger.info(f"Best parameters found: {study.best_params}")

    # Get best parameters
    best_parameters = study.best_params

    # Set the number of iterations as the best iteration
    # best_parameters["iterations"] = study.best_trial.user_attrs["best_iter"]

    return best_parameters


def train_final_model(
    x_training: pd.DataFrame,
    y_training: pd.Series,
    model_params: ModelParams,
    best_params: dict[str, Any],
    experiment_id: str,
) -> str:
    """
    Train the final model on the entire training set and evaluate it on the test set.

    Args:
        x_training (pd.DataFrame): The training features.
        y_training (pd.Series): The target labels for training.
        model_params (ModelParams): The parameters for the model.
        best_params (dict[str, Any]): The best hyperparameters found during the optimization.
        experiment_id (str): The id of the MLflow experiment.

    Returns:
        (str): The model ID in the MLflow tracking.
    """
    # Set the MLflow run
    tags = {"model_name": "final_model", "model_type": model_params.model_name}
    with mlflow.start_run(experiment_id=experiment_id, tags=tags):
        # Define the model pipeline
        pipe = _create_pipeline_model(
            model_name=model_params.model_name,
            n_comp=model_params.pca_n_components,
            params=best_params,
        )
        # Train the model
        pipe = _train_model_pipeline(pipe, x_training, y_training)
        y_pred_training = pipe.predict(x_training)

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
        model = mlflow.sklearn.log_model(pipe, name="model", input_example=x_training.sample(5))

        # Log the parameters
        mlflow.log_params(pipe.get_params())

    return model.model_id
