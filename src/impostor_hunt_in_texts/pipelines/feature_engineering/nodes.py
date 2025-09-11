"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.0.0
"""
# =================
# ==== IMPORTS ====
# =================

import logging
import re
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, concatenate_datasets
from nltk.tokenize import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from impostor_hunt_in_texts.pipelines.feature_engineering.validate_params import (
    ValidateParams,
)

# Options
logger = logging.getLogger(__name__)

# ===================
# ==== FUNCTIONS ====
# ===================


def validate_input_params(
    hf_model_name: str,
    max_length: int,
    stride: int,
    device: str,
) -> None:
    """Validate the input parameters for the feature engineering pipeline."""
    ValidateParams(
        hf_model_name=hf_model_name,
        max_length=max_length,
        stride=stride,
        device=device,
    )


def load_model_and_tokenizer(
    model_name: str,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
    """
    Load a pre-trained model and tokenizer from Hugging Face.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        (tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]): The pre-trained model and tokenizer.
    """
    return (
        transformers.AutoModel.from_pretrained(model_name),
        transformers.AutoTokenizer.from_pretrained(model_name),
    )


def augment_data(dataset: Dataset) -> Dataset:
    """Perform data augmentation by swaping places of texts.

    Args:
        dataset (Dataset): The dataset containing text pairs.

    Returns:
        (Dataset): The dataset with data augmentation.
    """
    text1, text2 = dataset["text1"], dataset["text2"]
    real_text_id = np.ones(len(text1)) * 3 - np.array(dataset["real_text_id"])
    dataset_swap = Dataset.from_dict({
        "text1": text2,
        "text2": text1,
        "real_text_id": real_text_id.astype(int),
    })
    return concatenate_datasets([dataset, dataset_swap], axis=0)


def _extract_mean_pooling_vector(  # noqa: PLR0913
    text: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model,
    max_length: int = 512,
    stride: int = 256,
    device: str = "cpu",
):
    """
    Extract mean pooling vector from text for a potentially large text using sliding window.

    Args:
        text (str): The input text to process.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for encoding the text.
        model (transformers.PreTrainedModel): The model to use for generating embeddings.
        max_length (int): Maximum length of the input sequence. Default is 512.
        stride (int): Stride for the sliding window. Default is 256.
        device (str): Device to run the model on ("cpu" or "cuda"). Default is "cpu".

    Returns:
        (np.ndarray): The mean pooling vector of the text.
    """
    encoded_text = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
    )

    input_ids_chunks = encoded_text["input_ids"]
    attention_mask_chunks = encoded_text["attention_mask"]

    all_mean_vecs = []

    # Options of the model
    model.to(device)
    model.eval()

    with torch.no_grad():
        for input_ids, attention_mask in zip(input_ids_chunks, attention_mask_chunks):
            input_ids = input_ids.unsqueeze(0).to(device)  # noqa: PLW2901
            attention_mask = attention_mask.unsqueeze(0).to(device)  # noqa: PLW2901

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = (
                outputs.last_hidden_state
            )  # shape: [1, seq_len, hidden_dim]

            # Apply mean pooling (excluding padded tokens)
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            masked_hidden = last_hidden_state * mask
            summed = masked_hidden.sum(dim=1)
            count = mask.sum(dim=1)
            mean_vec = summed / count

            all_mean_vecs.append(mean_vec.squeeze(0))

    # Average over all chunks to form the final vector
    final_vec = torch.stack(all_mean_vecs).mean(dim=0)

    return final_vec.cpu()


def extract_features(  # noqa: PLR0913
    dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    max_length: int = 512,
    stride: int = 256,
    device: str = "cpu",
) -> tuple[np.ndarray, list[int]]:
    """
    Extract interaction-based features from each text pair.

    Args:
        dataset (Dataset): The dataset containing text pairs.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use to encode the text.
        model (transformers.PreTrainedModel): The model to use to generate embeddings.
        max_length (int): Maximum length of the input sequence. Default is 512.
        stride (int): Stride for the sliding window. Default is 256.
        device (str): Device to run the model on ("cpu" or "cuda"). Default is "cpu".

    Returns:
        features (np.ndarray): Features extracted from the text pairs.
        df_ids (pd.DataFrame): DataFrame containing the IDs and real_text_ids (if available).
    """
    features = []
    ids = []
    real_text_ids = []

    for row in tqdm(dataset, desc="Extracting features"):
        vec1 = _extract_mean_pooling_vector(
            row["text1"], tokenizer, model, max_length, stride, device
        )
        vec2 = _extract_mean_pooling_vector(
            row["text2"], tokenizer, model, max_length, stride, device
        )

        # Compute interaction vectors
        diff = vec1 - vec2
        prod = vec1 * vec2
        ratio = vec1 / (vec2 + 1e-6)
        # cos_similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        # eucl_distance = np.linalg.norm(vec1 - vec2)

        # Concatenate all parts
        final_vec = torch.cat([vec1, vec2, diff, prod, ratio])
        features.append(final_vec.numpy())
        ids.append(row["id"])
        if "real_text_id" in row:
            real_text_ids.append(row["real_text_id"])

    if len(real_text_ids) > 0:
        df_ids = pd.DataFrame({"id": ids, "real_text_id": real_text_ids})
    else:
        logger.warning("The real_text_id column is not present in the dataset.")
        df_ids = pd.DataFrame({"id": ids})
    return np.array(features), df_ids


def convert_features_to_dataframe(
    dataset_features: np.ndarray, df_ids: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert the features extracted from the texts to a pandas DataFrame.

    Args:
        dataset_features (np.array): The features extracted from the text using the huggingface model.
        df_ids (pd.DataFrame): DataFrame containing the IDs and real_text_ids (if available).

    Returns:
        (pd.DataFrame): A DataFrame containing the features with columns for each feature and the ids.
    """
    return pd.concat(
        [
            df_ids,
            pd.DataFrame(
                dataset_features,
                columns=[f"token_feat_{i}" for i in range(dataset_features.shape[1])],
            ),
        ],
        axis=1,
    )


def _extract_text_info(text: str):
    """Extract information from the text.

    Args:
        text (str): The text from which info is extracted.

    Returns:
        features (dict): A dictionary containing the extracted features.
    """
    text = re.sub(r"\s+", " ", text).strip()
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    features = {
        "char_count": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": np.mean([len(word) for word in words]),
    }
    non_space_chars = list(re.findall(r"\S", text))
    if non_space_chars:
        latin_chars = [c for c in non_space_chars if 'LATIN' in unicodedata.name(c, '')]
        features["latin_ratio"] = len(latin_chars) / len(non_space_chars)
    else:
        features["latin_ratio"] = 0.0
    return features


def create_differential_features(df: pd.DataFrame, model_to_load: Optional[str] = None) -> pd.DataFrame:
    """Create differential features between text 1 and text 2.

    Args:
        df (pd.DataFrame): The input DataFrame containing 'text1' and 'text2' columns.
        model_to_load (str): The model to load and to use for embedding and similarity calculation.

    Returns:
        df (pd.DataFrame): The DataFrame with new differential features added.
    """
    # Load the model if specified
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformer model: {e}")
        sbert_model = None
    features_1 = df['text1'].apply(_extract_text_info).apply(pd.Series)
    features_2 = df['text2'].apply(_extract_text_info).apply(pd.Series)
    features_cols = list(features_1.columns)
    for col in tqdm(features_cols, desc="Creating differential features"):
        df[f"{col}_diff"] = features_1[col] - features_2[col]
        df[f"{col}_ratio"] = features_1[col] / (features_2[col] + 1e-6)
    if sbert_model is not None:
        logger.info("Calculating SBERT embeddings and cosine similarity.")
        embeddings_1 = sbert_model.encode(df["text1"].tolist(), show_progress_bar=True, batch_size=16)
        embeddings_2 = sbert_model.encode(df["text2"].tolist(), show_progress_bar=True, batch_size=16)
        df["cosine_similarity"] = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(embeddings_1, embeddings_2)]
        df["euclidean_distance"] = [np.linalg.norm(e1 - e2) for e1, e2 in zip(embeddings_1, embeddings_2)]
    final_features_cols = [
        f"{col}_diff" for col in features_cols
    ] + [
        f"{col}_ratio" for col in features_cols
    ]
    if "cosine_similarity" in df.columns:
        final_features_cols.extend(["cosine_similarity", "euclidean_distance"])
    for col in final_features_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0, 0).replace([np.inf, -np.inf], 0)
    return df
