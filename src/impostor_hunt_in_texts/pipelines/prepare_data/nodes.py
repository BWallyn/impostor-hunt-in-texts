"""This is a boilerplate pipeline 'prepare_data' generated using Kedro 0.19.14"""
# =================
# ==== IMPORTS ====
# =================

import os
import re

import pandas as pd
from datasets import Dataset, DatasetDict

from impostor_hunt_in_texts.pipelines.prepare_data.validate_params import ValidateParams

# ===================
# ==== FUNCTIONS ====
# ===================


def validate_input_params(
    path_data_train: str, path_data_test: str, path_dataset_dict: str
) -> None:
    """
    Validate the input parameters for the prepare_data pipeline.

    Args:
        path_data_train (str): Path to the training data directory.
        path_data_test (str): Path to the test data directory.
        path_dataset_dict (str): Path to the directory for saving the DatasetDict.

    Returns:
        None

    Raises:
        ValueError: If any of the paths do not match the expected pattern.
    """
    return ValidateParams(
        path_data_train=path_data_train,
        path_data_test=path_data_test,
        path_dataset_dict=path_dataset_dict,
    )


def _generate_dataset_train(df: pd.DataFrame, path_data: str):
    """
    Generate a dataset from the DataFrame and text files in the specified directory.

    Args:
        df (pd.DataFrame): DataFrame containing 'id', 'text1', and 'text2' columns.
        path_data (str): The path to the directory containing the text files.

    Returns:
        Yield dictionaries of (id, text1, text2).
    """
    for _, row in df.iterrows():
        folder_id = row["real_text_id"]
        folder_path = os.path.join(path_data, f"article_{folder_id:04d}")

        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()

        yield {"id": folder_id, "text1": text1, "text2": text2}


def _generate_dataset_test(path_data: str):
    """
    Generate a dataset from the text files in the specified directory.

    Args:
        path_data (str): The path to the directory containing the text files.

    Returns:
        Yield dictionaries of (text1, text2).
    """
    # Get list of folders matching the pattern "article_"
    folders = sorted(
        [
            f
            for f in os.listdir(path_data)
            if os.path.isdir(os.path.join(path_data, f)) and re.match(r"article_\d+", f)
        ]
    )

    # Get text files in each folder
    for folder in folders:
        folder_id = int(folder.split("_")[1])
        folder_path = os.path.join(path_data, folder)

        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()

        yield {"id": folder_id, "text1": text1, "text2": text2}


def create_dataset_train(df: pd.DataFrame, path_data: str) -> Dataset:
    """
    Create a Dataset for training from the DataFrame and text files.

    Args:
        df (pd.DataFrame): DataFrame containing 'id' and 'real_text_id' columns.
        path_data (str): The path to the directory containing the text files.

    Returns:
        (Dataset): A Dataset object containing the training data.
    """
    return Dataset.from_generator(lambda: _generate_dataset_train(df, path_data))


def create_dataset_test(path_data: str) -> Dataset:
    """
    Create a Dataset for test from text files.

    Args:
        path_data (str): The path to the directory containing the text files.

    Returns:
        (Dataset): A Dataset object containing the test data.
    """
    return Dataset.from_generator(lambda: _generate_dataset_test(path_data))


def create_datasets_dict(dataset_train: Dataset, dataset_test: Dataset) -> DatasetDict:
    """
    Create a DatasetDict containing train and test datasets.

    Args:
        dataset_train (Dataset): The train Dataset.
        dataset_test (Dataset): The test Dataset.

    Returns:
        (DatasetDict): A DatasetDict containing 'train' and 'test' datasets.
    """
    return DatasetDict(
        {
            "train": dataset_train,
            "test": dataset_test,
        }
    )
