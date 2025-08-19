"""
This is a boilerplate pipeline 'model_prediction'
generated using Kedro 1.0.0
"""
# =================
# ==== IMPORTS ====
# =================

import mlflow
import pandas as pd
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


def _make_prediction(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions using the trained model.

    Args:
        model (Pipeline): The trained model pipeline.
        df (pd.DataFrame): DataFrame containing the features for prediction.

    Returns:
        (np.array): Array of the predicted labels.
    """
    return model.predict(df)


def create_predictions_df(
    model: Pipeline, df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a DataFrame with predictions.

    Args:
        model (Pipeline): The trained model pipeline.
        df (pd.DataFrame): DataFrame containing the features for prediction.

    Returns:
        (pd.DataFrame): DataFrame with predictions.
    """
    predictions = _make_prediction(model, df)
    return pd.DataFrame({
        "id": df["id"],
        "real_text_id": predictions,
    })
