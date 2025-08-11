# =================
# ==== IMPORTS ====
# =================

from pydantic import BaseModel, Field, StrictStr

# ===============
# ==== CLASS ====
# ===============

class ValidateParams(BaseModel):
    """Class to validate the input parameters of the prepare_data pipeline."""

    path_data_train: StrictStr = Field(
        pattern=r"data\/[A-Za-z0-9\/\/]+",
        description="Path to the training data directory containing the text files.",
        frozen=True,
    )
    path_data_test: StrictStr = Field(
        pattern=r"data\/[A-Za-z0-9\/\/]+",
        description="Path to the test data directory containing the text files.",
        frozen=True,
    )
    path_dataset_dict: StrictStr = Field(
        pattern=r"data\/[A-Za-z0-9\/\/]+",
        description="Path to the data directory that will contain the Huggging Face DatasetDict.",
        frozen=True,
    )
