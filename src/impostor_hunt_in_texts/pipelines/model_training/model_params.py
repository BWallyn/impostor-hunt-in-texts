# =================
# ==== IMPORTS ====
# =================

from typing import Any

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
    search_params: dict[dict[str, Any]] = Field(
        default_factory=dict,
        description="The search parameters space for the bayesian optimization.",
        frozen=True,
    )
