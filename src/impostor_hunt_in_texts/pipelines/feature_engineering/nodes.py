"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.0.0
"""
# =================
# ==== IMPORTS ====
# =================

import numpy as np
import torch
import transformers
from datasets import Dataset
from tqdm import tqdm

from impostor_hunt_in_texts.pipelines.feature_engineering.validate_params import (
    ValidateParams,
)

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


def load_model_and_tokenizer(model_name: str) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
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


def _extract_mean_pooling_vector(  # noqa: PLR0913
    text: str,
    tokenizer : transformers.PreTrainedTokenizer,
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
        padding="max_length"
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
            last_hidden_state = outputs.last_hidden_state  # shape: [1, seq_len, hidden_dim]

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
) -> tuple[np.ndarray, list]:
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
        ids (list[str]): List of sample IDs.
    """
    features = []
    ids = []

    for row in tqdm(dataset, desc="Extracting features"):
        vec1 = _extract_mean_pooling_vector(row['text1'], tokenizer, model, max_length, stride, device)
        vec2 = _extract_mean_pooling_vector(row['text2'], tokenizer, model, max_length, stride, device)

        # Compute interaction vectors
        diff = vec1 - vec2
        prod = vec1 * vec2

        # Concatenate all parts
        final_vec = torch.cat([vec1, vec2, diff, prod])
        features.append(final_vec.numpy())
        ids.append(row['id'])

    return np.array(features), ids
