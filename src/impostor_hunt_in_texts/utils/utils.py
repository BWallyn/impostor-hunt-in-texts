# =================
# ==== IMPORTS ====
# =================

from datasets import DatasetDict

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
    return {key: f"{path_to_save}/{key}" for key in dataset_dict.keys()}
