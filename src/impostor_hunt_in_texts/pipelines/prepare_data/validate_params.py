# =================
# ==== IMPORTS ====
# =================

from pydantic import BaseModel, Field

# ===============
# ==== CLASS ====
# ===============

class ValidateParams(BaseModel):
    """Class to validate the input parameters of the prepare_data pipeline."""

    path_data_train: str = Field(
        pattern=r"data\/[A-Za-z0-9\/\/]+",
        description="Path to the training data directory containing the text files.",
        frozen=True,
    )
    path_data_test: str = Field(
        pattern=r"data\/[A-Za-z0-9\/\/]+",
        description="Path to the test data directory containing the text files.",
        frozen=True,
    )
    path_dataset_dict: str = Field(
        pattern=r"data\/[A-Za-z0-9\/\/]+",
        description="Path to the data directory that will contain the Huggging Face DatasetDict.",
        frozen=True,
    )
