# =================
# ==== IMPORTS ====
# =================

from typing import Any

from pydantic import BaseModel, Field, StrictStr

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
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the model. The parameters depend on the model type.",
        frozen=True
    )
