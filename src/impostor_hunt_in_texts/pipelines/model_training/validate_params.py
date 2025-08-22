# =================
# ==== IMPORTS ====
# =================

from typing import Any, Optional

from pydantic import BaseModel, Field, StrictInt, StrictStr

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
    model_pca_n_components: StrictInt = Field(
        ge=1,
        description="Number of components to get from the PCA.",
        frozen=True,
    )
    n_trials: StrictInt = Field(
        description="Number of trials for the bayesian optimization.",
        frozen=True,
    )
    default_hyperparameters: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Default hyperparameters to try during the bayesian optimization.",
        frozen=True,
    )
    search_space: dict[str, dict[str, Any]] = Field(
        description="Search space of the hyperparameters for the model.",
        frozen=True,
    )
    label_column: StrictStr = Field(
        pattern=r"^[A-Za-z0-9\_]+$",
        description="Name of the column containing the labels.",
    )
