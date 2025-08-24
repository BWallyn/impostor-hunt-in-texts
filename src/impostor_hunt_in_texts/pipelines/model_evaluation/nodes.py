"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 1.0.0
"""
# =================
# ==== IMPORTS ====
# =================

from typing import Optional

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ===================
# ==== FUNCTIONS ====
# ===================

def get_feature_importance(
    model: Pipeline,
    df: pd.DataFrame,
    dict_tables: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Get feature importance from the trained model.

    Args:
        model (Pipeline): The trained model pipeline.
        df (pd.DataFrame): DataFrame used to extract feature names.
        dict_tables (Optional[dict[str, pd.DataFrame]]): Dictionary to store the feature importance DataFrame. If None, a new dictionary will be created.

    Returns:
        (pd.DataFrame): DataFrame with feature importance.
    """
    # Options
    if dict_tables is None:
        dict_tables = {}

    # Get feature importance
    model_classif = model.named_steps["classifier"]
    if model_classif is RandomForestClassifier:
        importances = model_classif.feature_importances_
        dict_tables["model_evaluation/feature_importance"] = pd.DataFrame({
            "feature": df.columns,
            "importance": importances,
        }).sort_values(by="importance", ascending=False)
    return dict_tables


def log_tables_to_mlflow(dict_tables: dict[str, pd.DataFrame], run_id: str) -> None:
    """Log tables to MLflow.

    Args:
        dict_tables (dict[str, pd.DataFrame]): Dictionary containing the tables to log.
        run_id (str): The MLflow run ID.

    Returns:
        None
    """
    for table_name, df in dict_tables.items():
        mlflow.log_table(df, table_name, run_id=run_id)
