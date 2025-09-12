# =================
# ==== IMPORTS ====
# =================

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from impostor_hunt_in_texts.pipelines.feature_engineering.nodes import (
    # _extract_mean_pooling_vector,
    convert_features_to_dataframe,
    extract_features,
    load_model_and_tokenizer,
)
from transformers import PreTrainedModel, PreTrainedTokenizer

# ===============
# ==== TESTS ====
# ===============

# ==== Load model and tokenizer ====

def test_load_model_and_tokenizer_success():
    """Test the successful loading of a model and tokenizer."""
    # Mock the AutoModel and AutoTokenizer
    mock_model = MagicMock(spec=PreTrainedModel)
    mock_tokenizer = MagicMock(spec=PreTrainedTokenizer)

    with patch(
        "transformers.AutoModel.from_pretrained",
        return_value=mock_model,
    ) as mock_model_from_pretrained, patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    ) as mock_tokenizer_from_pretrained:
        # Call the function
        model, tokenizer = load_model_and_tokenizer("bert-base-uncased")
        # Assertions
        assert model == mock_model
        assert tokenizer == mock_tokenizer
        mock_model_from_pretrained.assert_called_once_with("bert-base-uncased")
        mock_tokenizer_from_pretrained.assert_called_once_with("bert-base-uncased")

def test_load_model_and_tokenizer_error():
    """Test the error handling when loading a model and tokenizer."""
    # Mock an exception for AutoModel
    with patch(
        "transformers.AutoModel.from_pretrained",
        side_effect=Exception("Model not found"),
    ), patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=MagicMock(spec=PreTrainedTokenizer),
    ):
        # Assert that the correct exception is raised
        with pytest.raises(Exception, match="Model not found"):
            load_model_and_tokenizer("non-existent-model")

def test_load_model_and_tokenizer_tokenizer_error():
    """Test the error handling when loading a tokenizer."""
    # Mock an exception for AutoTokenizer
    with patch(
        "transformers.AutoModel.from_pretrained",
        return_value=MagicMock(spec=PreTrainedModel),
    ), patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=Exception("Tokenizer not found"),
    ):
        # Assert that the correct exception is raised
        with pytest.raises(Exception, match="Tokenizer not found"):
            load_model_and_tokenizer("non-existent-model")


# ==== extract mean pooling vector ====

#TODO: Fix the tests
# def test_extract_mean_pooling_vector_basic():
#     """Test the extraction of mean pooling vector from text."""
#     # Mock tokenizer and model
#     tokenizer = MagicMock(spec=PreTrainedTokenizer)
#     model = MagicMock()

#     # Mock encoded output
#     mock_input_ids = torch.tensor([[1, 2, 3, 0, 0]])
#     mock_attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
#     tokenizer.return_value = {
#         "input_ids": [mock_input_ids],
#         "attention_mask": [mock_attention_mask],
#     }

#     # Mock model output
#     mock_last_hidden = torch.randn(1, 5, 10)  # [batch, seq_len, hidden_dim]
#     model.return_value.last_hidden_state = mock_last_hidden

#     # Call the function
#     result = _extract_mean_pooling_vector(
#         text="test text",
#         tokenizer=tokenizer,
#         model=model,
#         max_length=512,
#         stride=256,
#         device="cpu",
#     )

#     # Assertions
#     assert isinstance(result, torch.Tensor)
#     assert result.shape[0] == 10  # hidden_dim
#     tokenizer.assert_called_once_with(
#         "test text",
#         return_tensors="pt",
#         truncation=True,
#         max_length=5,
#         stride=3,
#         return_overflowing_tokens=True,
#         padding="max_length",
#     )
#     model.assert_called_once()
#     assert torch.allclose(result, mock_last_hidden[:, :3, :].mean(dim=(0, 1)))

# def test_extract_mean_pooling_vector_multiple_chunks():
#     """Test the extraction of mean pooling vector from text with multiple chunks."""
#     # Mock tokenizer and model
#     tokenizer = MagicMock(spec=PreTrainedTokenizer)
#     model = MagicMock()

#     # Mock encoded output with two chunks
#     mock_input_ids = [torch.tensor([[1, 2, 3, 0, 0]]), torch.tensor([[4, 5, 6, 0, 0]])]
#     mock_attention_mask = [torch.tensor([[1, 1, 1, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]])]
#     tokenizer.return_value = {
#         "input_ids": mock_input_ids,
#         "attention_mask": mock_attention_mask,
#     }

#     # Mock model output for each chunk
#     mock_last_hidden1 = torch.randn(1, 5, 10)
#     mock_last_hidden2 = torch.randn(1, 5, 10)
#     model.side_effect = [
#         MagicMock(last_hidden_state=mock_last_hidden1),
#         MagicMock(last_hidden_state=mock_last_hidden2),
#     ]

#     # Call the function
#     result = _extract_mean_pooling_vector(
#         text="long test text",
#         tokenizer=tokenizer,
#         model=model,
#         max_length=5,
#         stride=3,
#         device="cpu",
#     )

#     # Assertions
#     assert isinstance(result, torch.Tensor)
#     assert result.shape[0] == 10
#     assert model.call_count == 2

# def test_extract_mean_pooling_vector_empty_text():
#     """Test the extraction of mean pooling vector from an empty text."""
#     # Mock tokenizer and model
#     tokenizer = MagicMock(spec=PreTrainedTokenizer)
#     model = MagicMock()

#     # Mock encoded output for empty text
#     mock_input_ids = torch.tensor([[0, 0, 0, 0, 0]])
#     mock_attention_mask = torch.tensor([[0, 0, 0, 0, 0]])
#     tokenizer.return_value = {
#         "input_ids": [mock_input_ids],
#         "attention_mask": [mock_attention_mask],
#     }

#     # Mock model output
#     mock_last_hidden = torch.randn(1, 5, 10)
#     model.return_value.last_hidden_state = mock_last_hidden

#     # Call the function
#     result = _extract_mean_pooling_vector(
#         text="",
#         tokenizer=tokenizer,
#         model=model,
#         max_length=5,
#         stride=3,
#         device="cpu",
#     )

#     # Assertions
#     assert torch.isnan(result).any()  # Division by zero

# ==== Extract features ====

def test_extract_features_basic():
    """Test the extraction of features from a basic dataset."""
    # Mock dataset
    mock_dataset = [
        {"id": 1, "real_text_id": 1, "text1": "hello world", "text2": "world hello"},
        {"id": 2, "real_text_id": 2, "text1": "foo bar", "text2": "bar foo"},
    ]

    # Mock tokenizer and model
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    model = MagicMock(spec=PreTrainedModel)

    # Mock _extract_mean_pooling_vector to return fixed vectors
    mock_vec1 = torch.randn(10)
    mock_vec2 = torch.randn(10)
    with patch(
        "impostor_hunt_in_texts.pipelines.feature_engineering.nodes._extract_mean_pooling_vector",
        side_effect=[mock_vec1, mock_vec2, mock_vec1, mock_vec2],
    ) as mock_extract:

        # Call the function
        features, ids = extract_features(
            dataset=mock_dataset,
            tokenizer=tokenizer,
            model=model,
            max_length=512,
            stride=256,
            device="cpu",
        )

        # Assertions
        assert len(features) == 2
        assert len(ids) == 2
        # assert ids == [1, 2]
        assert features.shape == (2, 50)  # 5 * 10 (vec1, vec2, diff, prod, ratio)
        mock_extract.assert_called_with(
            mock_dataset[1]["text2"], tokenizer, model, 512, 256, "cpu"
        )

        # Check feature construction
        expected_vec = torch.cat([mock_vec1, mock_vec2, mock_vec1 - mock_vec2, mock_vec1 * mock_vec2])
        assert torch.allclose(torch.tensor(features[0]), expected_vec, atol=1e-6)

def test_extract_features_empty_dataset():
    """Test the extraction of features from an empty dataset."""
    # Mock empty dataset
    mock_dataset = []

    # Mock tokenizer and model
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    model = MagicMock(spec=PreTrainedModel)

    # Call the function
    features, ids = extract_features(
        dataset=mock_dataset,
        tokenizer=tokenizer,
        model=model,
        max_length=512,
        stride=256,
        device="cpu",
    )

    # Assertions
    assert len(features) == 0
    assert len(ids) == 0

def test_extract_features_device():
    """Test the extraction of features with a specific device."""
    # Mock dataset
    mock_dataset = [{"id": 1, "text1": "hello", "text2": "world"}]

    # Mock tokenizer and model
    tokenizer = MagicMock(spec=PreTrainedTokenizer)
    model = MagicMock(spec=PreTrainedModel)

    # Mock _extract_mean_pooling_vector
    mock_vec1 = torch.randn(10)
    mock_vec2 = torch.randn(10)
    with patch(
        "impostor_hunt_in_texts.pipelines.feature_engineering.nodes._extract_mean_pooling_vector",
        side_effect=[mock_vec1, mock_vec2],
    ) as mock_extract:

        # Call the function with cuda
        with patch("torch.cuda.is_available", return_value=True):
            features, ids = extract_features(
                dataset=mock_dataset,
                tokenizer=tokenizer,
                model=model,
                max_length=512,
                stride=256,
                device="cuda",
            )

            # Assertions
            assert len(features) == 1
            assert len(ids) == 1
            mock_extract.assert_called_with(
                mock_dataset[0]["text2"], tokenizer, model, 512, 256, "cuda"
            )


# ==== Convert features to dataframe ====

def test_convert_features_to_dataframe_basic():
    """Test the conversion of features to a dataframe."""
    # Mock features and ids
    dataset_features = np.array([
        [1.1, 2.2, 3.3],
        [4.4, 5.5, 6.6],
    ])
    ids = pd.DataFrame({"id": [101, 102]})

    # Call the function
    df = convert_features_to_dataframe(dataset_features, ids)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 4)  # 2 rows, 1 id column + 3 feature columns
    assert list(df.columns) == ["id", "token_feat_0", "token_feat_1", "token_feat_2"]
    # assert list(df["id"]) == ids
    assert list(df["token_feat_0"]) == [1.1, 4.4]
    assert list(df["token_feat_1"]) == [2.2, 5.5]
    assert list(df["token_feat_2"]) == [3.3, 6.6]

def test_convert_features_to_dataframe_empty():
    """Test the conversion of features to a dataframe with empty numpy array."""
    # Mock empty features and ids
    dataset_features = np.array([])
    dataset_features.resize((0, 3))  # Empty 2D array with 3 columns
    ids = pd.DataFrame([])

    # Call the function
    df = convert_features_to_dataframe(dataset_features, ids)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (0, 3)
    assert list(df.columns) == ["token_feat_0", "token_feat_1", "token_feat_2"]

def test_convert_features_to_dataframe_single_feature():
    """Test the conversion of features to a dataframe with a single feature."""
    # Mock features and ids with a single feature
    dataset_features = np.array([
        [1.1],
        [4.4],
    ])
    ids = pd.DataFrame({"id": [101, 102]})

    # Call the function
    df = convert_features_to_dataframe(dataset_features, ids)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)  # 2 rows, 1 id column + 1 feature column
    assert list(df.columns) == ["id", "token_feat_0"]
    # assert list(df["id"]) == ids
    assert list(df["token_feat_0"]) == [1.1, 4.4]
