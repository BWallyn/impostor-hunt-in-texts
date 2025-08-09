"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline

from impostor_hunt_in_texts.pipelines.feature_engineering.nodes import (
    extract_features,
    load_model_and_tokenizer,
)
from impostor_hunt_in_texts.utils.utils import load_hf_datasetdict, split_dataset_dict


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline."""
    return Pipeline(
        nodes=[
            Node(
                func=load_hf_datasetdict,
                inputs="dict_metadata_datasets",
                outputs="dataset_dict",
                name="Load_dataset_dict",
            ),
            Node(
                func=split_dataset_dict,
                inputs="dataset_dict",
                outputs=["dataset_train", "dataset_test"],
                name="Split_dataset_dict",
            ),
            Node(
                func=load_model_and_tokenizer,
                inputs={"model_name": "params:hf_model_name"},
                outputs=["model", "tokenizer"],
                name="Load_model_and_tokenizer",
            ),
            Node(
                func=extract_features,
                inputs={
                    "dataset": "dataset_train",
                    "tokenizer": "tokenizer",
                    "model": "model",
                    "max_length": "params:max_length",
                    "stride": "params:stride",
                    "device": "params:device",
                },
                outputs=["dataset_train_features", "train_ids"],
                name="Extract_features_text_train",
            ),
            Node(
                func=extract_features,
                inputs={
                    "dataset": "dataset_test",
                    "tokenizer": "tokenizer",
                    "model": "model",
                    "max_length": "params:max_length",
                    "stride": "params:stride",
                    "device": "params:device",
                },
                outputs=["dataset_test_features", "test_ids"],
                name="Extract_features_text_test",
            ),
        ],
        namespace="feature_engineering",
        inputs="dict_metadata_datasets",
    )
