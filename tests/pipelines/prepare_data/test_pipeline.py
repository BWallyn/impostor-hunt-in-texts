"""
This is a boilerplate test file for pipeline 'prepare_data'
generated using Kedro 0.19.14.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
import logging

import pandas as pd
from impostor_hunt_in_texts.pipelines.prepare_data.pipeline import (
    create_pipeline as create_pd_pipeline,
)
from kedro.io import DataCatalog
from kedro.runner import SequentialRunner


def test_prepare_data_pipeline(caplog):
    """Test the prepare data pipeline."""
    # Arrange pipeline
    pipeline = (
        create_pd_pipeline().
        from_nodes(
            "prepare_data.Validate_input_parameters_prepare_data",
            "prepare_data.Create_dataset_train",
            "prepare_data.Create_dataset_test",
        )
        .to_nodes("prepare_data.Create_datasets_dict")
    )

    # Arrange data catalog
    catalog = DataCatalog()

    dummy_data = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "real_text_id": [1, 2, 1, 2],
        }
    )

    dummy_parameters = {
        "path_data_train": "tests/data/train",
        "path_data_test": "tests/data/test",
        "path_dataset_dict": "tests/data/dict_datasets",
    }

    catalog["df_train"] = dummy_data
    catalog["params:prepare_data.path_data_train"] = dummy_parameters["path_data_train"]
    catalog["params:prepare_data.path_data_test"] = dummy_parameters["path_data_test"]
    catalog["params:prepare_data.path_dataset_dict"] = dummy_parameters["path_dataset_dict"]

    # Arrange the log testing setup
    caplog.set_level(logging.DEBUG, logger="kedro") # Ensure all logs produced by Kedro are captured
    successful_run_msg = "Pipeline execution completed successfully"

    # Act
    SequentialRunner().run(pipeline, catalog)

    # Assert
    assert successful_run_msg in caplog.text
