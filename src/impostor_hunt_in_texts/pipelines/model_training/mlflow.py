# =================
# ==== IMPORTS ====
# =================

from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import mlflow

# Options
VERSION_FORMAT = "%Y_%m_%dT%H_%M_%S_%fZ"

# ===================
# ==== FUNCTIONS ====
# ===================


def _generate_timestamp() -> str:
    """
    Generate timestamp to be used by versionning

    Args:
        None

    Returns:
        (str): String representation of the current timestamp
    """
    current_ts = datetime.now(tz=UTC).strftime(VERSION_FORMAT)
    return current_ts[:-4] + current_ts[-1:]  # Dont keep microseconds


def create_mlflow_experiment(
    experiment_folder_path: str,
    experiment_name: str,
    experiment_id: Optional[str] = None,
) -> str:
    """
    Create a MLflow experiment or log it if it already exists.

    Args:
        experiment_folder_path (str): Path to the MLflow experiment folder
        experiment_name (str): Name of the experiment
        experiment_id (Optional[str]): Id of the experiment if it already exists.

    Returns:
        (str): Id of the created experiment
    """
    if experiment_id is None:
        # Create MLflow experiment
        time = _generate_timestamp()
        experiment_id = mlflow.create_experiment(
            name=f"{experiment_name}_{time}",
            artifact_location=Path.cwd().joinpath(experiment_folder_path).as_uri(),
        )
    return experiment_id
