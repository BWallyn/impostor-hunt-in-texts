# =================
# ==== IMPORTS ====
# =================

from unittest.mock import MagicMock, patch

import pytest
from impostor_hunt_in_texts.pipelines.feature_engineering.nodes import (
    # _extract_mean_pooling_vector,
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
