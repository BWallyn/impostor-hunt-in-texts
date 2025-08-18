# =================
# ==== IMPORTS ====
# =================

from typing import Any

from pydantic import BaseModel, Field, StrictStr

# ===============
# ==== CLASS ====
# ===============

class ValidateParams(BaseModel):
    """Class to validate the input parameters of the feature_engineering pipeline."""

    experiment_folder_path: StrictStr = Field(
        pattern=r"^[A-Za-z0-9\-_\/\.]+$",
        description="MLflow experiment folder path.",
        frozen=True,
    )
    experiment_name: StrictStr = Field(
        pattern=r"^[A-Za-z0-9\_]+$",
        description="MLflow experiment name.",
        frozen=True,
    )
    experiment_id_saved: StrictStr = Field(
        pattern=r"^[A-Za-z0-9]+$",
        description="MLflow experiment id if it exists.",
        frozen=True,
    )
    model_name: StrictStr = Field(
        pattern=r"^(RandomForestClassifier|HistGradientBoostingClassifier)$",
        description="Type of model to train.",
        frozen=True,
    )
    model_params: dict[str, Any] = Field(
        description="Parameters for the model.",
        frozen=True,
    )
    label_column: StrictStr = Field(
        pattern=r"^[A-Za-z0-9\_]+$",
        description="Name of the column containing the labels."
    )
