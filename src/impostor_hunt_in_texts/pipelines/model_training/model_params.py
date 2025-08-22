# =================
# ==== IMPORTS ====
# =================

from typing import Any, Optional

from pydantic import BaseModel, Field, StrictInt, StrictStr

# ===============
# ==== CLASS ====
# ===============

class ModelParams(BaseModel):
    """Class to store the model parameters."""

    model_name: StrictStr = Field(
        pattern=r"^(RandomForestClassifier|HistGradientBoosting)$",
        description="Name of the model to use for training.",
        frozen=True,
    )
    pca_n_components: StrictInt = Field(
        ge=1,
        description="Number of components to get from the PCA.",
        frozen=True,
    )
    default_hyperparameters: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Default hyperparameters to try during the bayesian optimization.",
        frozen=True,
    )
    search_params: dict[str, dict[str, Any]] = Field(
        description="The search parameters space for the bayesian optimization.",
        frozen=True,
    )
    n_trials: StrictInt = Field(
        default=50,
        description="Number of trials for the bayesian optimization.",
        frozen=True,
    )
