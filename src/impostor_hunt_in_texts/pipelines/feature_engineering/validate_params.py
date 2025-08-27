# =================
# ==== IMPORTS ====
# =================

from pydantic import BaseModel, Field, StrictInt, StrictStr

# ===============
# ==== CLASS ====
# ===============


class ValidateParams(BaseModel):
    """Class to validate the input parameters of the feature_engineering pipeline."""

    hf_model_name: StrictStr = Field(
        pattern=r"^[A-Za-z0-9\-_\/]+$",
        description="Name of the Hugging Face model to use for feature extraction.",
        frozen=True,
    )
    max_length: StrictInt = Field(
        ge=16,
        description="Maximum length of the input sequence for the model.",
        frozen=True,
    )
    stride: StrictInt = Field(
        ge=16,
        description="Stride for the sliding window during feature engineering.",
        frozen=True,
    )
    device: StrictStr = Field(
        pattern=r"^(cpu|cuda)$",
        description="Device to run the model on ('cpu' or 'cuda').",
        frozen=True,
    )
