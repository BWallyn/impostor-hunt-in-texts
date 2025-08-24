# =================
# ==== IMPORTS ====
# =================

import mlflow
from sklearn.pipeline import Pipeline

# ===================
# ==== FUNCTIONS ====
# ===================

def load_model(model_id: str) -> Pipeline:
    """
    Load the trained model from MLflow.

    Args:
        model_id (str): The ID of the model in MLflow.

    Returns:
        (Pipeline): The loaded model pipeline.
    """
    return mlflow.pyfunc.load_model(f"models:/{model_id}")


def load_model_sklearn(model_id: str) -> Pipeline:
    """Load the trained scikit-learn model from MLflow.

    Args:
        model_id (str): The ID of the model in MLflow.

    Returns:
        (Pipeline): The loaded scikit-learn model pipeline.
    """
    return mlflow.sklearn.load_model(f"models:/{model_id}")
