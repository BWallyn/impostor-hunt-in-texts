"""
This is a boilerplate pipeline 'prepare_data'
generated using Kedro 0.19.14
"""
# =================
# ==== IMPORTS ====
# =================

import os
import pandas as pd


# ===================
# ==== FUNCTIONS ====
# ===================

def generate_dataset_train(df: pd.DataFrame, path_data: str):
    """Generate a dataset from the DataFrame and text files in the specified directory.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'id', 'text1', and 'text2' columns.
        path_data (str): The path to the directory containing the text files.

    Returns:
        Yield dictionaries of (id, text1, text2).
    """
    for _, row in df.iterrows():
        folder_id = row['real_text_id']
        folder_path = os.path.join(path_data, f"article_{folder_id:04d}")

        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()
    
        yield {
            "id": folder_id,
            "text1": text1,
            "text2": text2
        }


def generate_dataset_test(path_data: str):
    """Generate a dataset from the text files in the specified directory.
    
    Args:
        path_data (str): The path to the directory containing the text files.

    Returns:
        Yield dictionaries of (text1, text2).
    """
    # Get list of folders matching the pattern "article_"
    folders = sorted([
        f for f in os.listdir(path_data)
        if os.path.isdir(os.path.join(path_data, f)) and re.match(r'article_\d+', f)
    ])

    # Get text files in each folder
    for folder in folders:
        folder_id = int(folder.split('_')[1])
        folder_path = os.path.join(path_data, folder)

        file1_path = os.path.join(folder_path, "file_1.txt")
        file2_path = os.path.join(folder_path, "file_2.txt")

        with open(file1_path, encoding="utf-8") as f1:
            text1 = f1.read()
        with open(file2_path, encoding="utf-8") as f2:
            text2 = f2.read()
    
        yield {
            "id": folder_id,
            "text1": text1,
            "text2": text2
        }