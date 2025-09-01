# =================
# ==== IMPORTS ====
# =================

import os
import shutil
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
from impostor_hunt_in_texts.pipelines.prepare_data.nodes import _generate_dataset_train

# ===============
# ==== TESTS ====
# ===============

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'real_text_id': [1, 2, 1],
        'other_column': ['a', 'b', 'c']
    })

@pytest.fixture
def temp_directory():
    """Create a temporary directory with test files."""
    temp_dir = tempfile.mkdtemp()

    # Create test directories and files
    for folder_id in [1, 2, 3]:
        folder_path = os.path.join(temp_dir, f"article_{folder_id:04d}")
        os.makedirs(folder_path, exist_ok=True)

        # Create test files
        with open(os.path.join(folder_path, "file_1.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Content of file 1 for article {folder_id}")
        with open(os.path.join(folder_path, "file_2.txt"), 'w', encoding='utf-8') as f:
            f.write(f"Content of file 2 for article {folder_id}")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)

def test_generate_dataset_train_success(sample_dataframe, temp_directory):
    """Test successful generation of dataset."""
    results = list(_generate_dataset_train(sample_dataframe, temp_directory))

    assert len(results) == 3

    # Check first result
    assert results[0]['id'] == 1
    assert results[0]['text1'] == "Content of file 1 for article 1"
    assert results[0]['text2'] == "Content of file 2 for article 1"

    # Check second result
    assert results[1]['id'] == 2
    assert results[1]['text1'] == "Content of file 1 for article 2"
    assert results[1]['text2'] == "Content of file 2 for article 2"

    # Check third result
    assert results[2]['id'] == 3
    assert results[2]['text1'] == "Content of file 1 for article 3"
    assert results[2]['text2'] == "Content of file 2 for article 3"

def test_generate_dataset_train_empty_dataframe(temp_directory):
    """Test with empty DataFrame."""
    empty_df = pd.DataFrame({'real_text_id': []})
    results = list(_generate_dataset_train(empty_df, temp_directory))

    assert len(results) == 0

def test_generate_dataset_train_single_row(temp_directory):
    """Test with single row DataFrame."""
    single_row_df = pd.DataFrame({'id': [1], 'real_text_id': [1]})
    results = list(_generate_dataset_train(single_row_df, temp_directory))

    assert len(results) == 1
    assert results[0]['id'] == 1
    assert results[0]['text1'] == "Content of file 1 for article 1"
    assert results[0]['text2'] == "Content of file 2 for article 1"

def test_generate_dataset_train_missing_file1(sample_dataframe, temp_directory):
    """Test behavior when file_1.txt is missing."""
    # Remove file_1.txt from first article
    file1_path = os.path.join(temp_directory, "article_0001", "file_1.txt")
    os.remove(file1_path)

    with pytest.raises(FileNotFoundError):
        list(_generate_dataset_train(sample_dataframe, temp_directory))

def test_generate_dataset_train_missing_file2(sample_dataframe, temp_directory):
    """Test behavior when file_2.txt is missing."""
    # Remove file_2.txt from first article
    file2_path = os.path.join(temp_directory, "article_0001", "file_2.txt")
    os.remove(file2_path)

    with pytest.raises(FileNotFoundError):
        list(_generate_dataset_train(sample_dataframe, temp_directory))

def test_generate_dataset_train_missing_directory(sample_dataframe, temp_directory):
    """Test behavior when article directory is missing."""
    # Remove entire directory for first article
    article_dir = os.path.join(temp_directory, "article_0001")
    shutil.rmtree(article_dir)

    with pytest.raises(FileNotFoundError):
        list(_generate_dataset_train(sample_dataframe, temp_directory))

def test_generate_dataset_train_nonexistent_path(sample_dataframe):
    """Test with non-existent data path."""
    nonexistent_path = "/path/that/does/not/exist"

    with pytest.raises(FileNotFoundError):
        list(_generate_dataset_train(sample_dataframe, nonexistent_path))

def test_generate_dataset_train_unicode_content(temp_directory):
    """Test with Unicode content in files."""
    # Create DataFrame with one row
    df = pd.DataFrame({'id': [1], 'real_text_id': [1]})

    # Create files with Unicode content
    folder_path = os.path.join(temp_directory, "article_0001")
    with open(os.path.join(folder_path, "file_1.txt"), 'w', encoding='utf-8') as f:
        f.write("Content with Unicode: café, naïve, 中文")
    with open(os.path.join(folder_path, "file_2.txt"), 'w', encoding='utf-8') as f:
        f.write("More Unicode: émoji 🚀, Ångström")

    results = list(_generate_dataset_train(df, temp_directory))

    assert len(results) == 1
    assert results[0]['text1'] == "Content with Unicode: café, naïve, 中文"
    assert results[0]['text2'] == "More Unicode: émoji 🚀, Ångström"

def test_generate_dataset_train_large_ids(temp_directory):
    """Test with large ID numbers to verify zero-padding."""
    df = pd.DataFrame({'id': [9999], 'real_text_id': [9999]})

    # Create corresponding directory and files
    folder_path = os.path.join(temp_directory, "article_9999")
    os.makedirs(folder_path, exist_ok=True)

    with open(os.path.join(folder_path, "file_1.txt"), 'w', encoding='utf-8') as f:
        f.write("Large ID content 1")
    with open(os.path.join(folder_path, "file_2.txt"), 'w', encoding='utf-8') as f:
        f.write("Large ID content 2")

    results = list(_generate_dataset_train(df, temp_directory))

    assert len(results) == 1
    assert results[0]['id'] == 9999
    assert results[0]['text1'] == "Large ID content 1"
    assert results[0]['text2'] == "Large ID content 2"

def test_generate_dataset_train_empty_files(temp_directory):
    """Test with empty text files."""
    df = pd.DataFrame({'id': [1], 'real_text_id': [1]})

    # Create empty files
    folder_path = os.path.join(temp_directory, "article_0001")
    with open(os.path.join(folder_path, "file_1.txt"), 'w', encoding='utf-8') as f:
        f.write("")
    with open(os.path.join(folder_path, "file_2.txt"), 'w', encoding='utf-8') as f:
        f.write("")

    results = list(_generate_dataset_train(df, temp_directory))

    assert len(results) == 1
    assert results[0]['text1'] == ""
    assert results[0]['text2'] == ""

def test_generate_dataset_train_is_generator(sample_dataframe, temp_directory):
    """Test that function returns a generator."""
    result = _generate_dataset_train(sample_dataframe, temp_directory)

    # Check that it's a generator
    assert hasattr(result, '__iter__')
    assert hasattr(result, '__next__')

    # Check that we can iterate through it
    first_item = next(result)
    assert isinstance(first_item, dict)
    assert 'id' in first_item
    assert 'text1' in first_item
    assert 'text2' in first_item

@patch('builtins.open', side_effect=PermissionError("Permission denied"))
def test_generate_dataset_train_permission_error(mock_open_func, sample_dataframe, temp_directory):
    """Test behavior when file access is denied."""
    with pytest.raises(PermissionError):
        list(_generate_dataset_train(sample_dataframe, temp_directory))

def test_generate_dataset_train_dataframe_missing_column(temp_directory):
    """Test with DataFrame missing required 'real_text_id' column."""
    df_missing_column = pd.DataFrame({'wrong_column': [1, 2, 3]})

    with pytest.raises(KeyError):
        list(_generate_dataset_train(df_missing_column, temp_directory))
