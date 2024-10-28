"""
extract_training_data.py

This module extracts the most recent training and validation data loaders from a ZenML pipeline. 
It is designed to retrieve the output of the "prepare_dataloaders" step from the latest run of 
the "training_data_pipeline" pipeline, making the data loaders available for further model 
training or evaluation tasks.

Functions:
    extract_latest_loaders: Retrieves the latest training and validation data loaders 
                            from the specified ZenML pipeline and run.

Usage:
    Use this module to access the latest data loaders for quick model evaluation or analysis. 
    The returned data loaders are in a dictionary format, allowing direct access to training 
    and validation batches.
"""

import os
import sys
import warnings

from zenml.client import Client


def extract_latest_loaders() -> dict:
    """Retrieves the latest training and validation data loaders from the ZenML pipeline.

    This function connects to the ZenML client, identifies the latest run of the 
    'training_data_pipeline', and extracts the 'prepare_dataloaders' step's output artifact.
    
    Returns:
        dict: A dictionary containing the most recent training and validation data loaders.
            {
                "train": DataLoader for the training set,
                "validation": DataLoader for the validation set
            }

    Prints:
        Confirmation message with the pipeline run ID upon successful extraction.
    """
    client = Client()
    pipline_name = "training_data_pipeline"
    pipeline = client.get_pipeline(pipline_name)
    latest_run = pipeline.last_run
    loaders_step = latest_run.steps["prepare_dataloaders"]
    artifact = loaders_step.output.load()
    train_loader = artifact["train"]
    val_loader = artifact["validation"]
    print(f"Pipeline artifact: {latest_run.id} loaded successfully")
    return {"train": train_loader, "validation": val_loader}
