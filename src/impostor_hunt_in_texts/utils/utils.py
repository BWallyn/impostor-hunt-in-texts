# =================
# ==== IMPORTS ====
# =================

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk

# ===================
# ==== FUNCTIONS ====
# ===================


def save_hf_datasetdict(dataset_dict: DatasetDict, path_to_save: str) -> dict[str, str]:
    """
    Save the Hugging Face DatasetDict and store all the infos in a dictionary.

    Args:
        dataset_dict (DatasetDict): The DatasetDict to save.
        path_to_save (str): The path where the DatasetDict will be saved.

    Returns:
        (dict[str, str]): A dictionary containing the paths to the saved datasets in dict.
    """
    dataset_dict.save_to_disk(path_to_save)
    return {"save_path": path_to_save}


def load_hf_datasetdict(dict_metadata_datasets: dict[str, str]) -> DatasetDict:
    """
    Load the Hugging Face DatasetDict from the model metadata dictionary.

    Args:
        dict_metadata_datasets (dict[str, str]): A dictionary containing the paths to the saved datasets.

    Returns:
        (DatasetDict): The loaded DatasetDict.
    """
    return load_from_disk(dict_metadata_datasets["save_path"])


def split_dataset_dict(dataset_dict: DatasetDict) -> tuple[Dataset, Dataset]:
    """
    Split the DatasetDict into train and test datasets.

    Args:
        dataset_dict (DatasetDict): The DatasetDict containing 'train' and 'test' datasets.

    Returns:
        (tuple[Dataset, Dataaset]): A tuple containing the train and test datasets.
    """
    return dataset_dict["train"], dataset_dict["test"]


def drop_columns(df: pd.DataFrame, cols_to_drop: list[str]) -> pd.DataFrame:
    """Drop specified columns from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols_to_drop (list[str]): List of the columns to drop.

    Returns:
        (pd.DataFrame): The DataFrame with the specified columns dropped.
    """
    return df.drop(columns=cols_to_drop)
