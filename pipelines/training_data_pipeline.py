"""
training_data_pipeline.py

This module defines a ZenML pipeline for preparing training and validation data 
from raw tweet data for a sentiment analysis model. The pipeline includes steps 
to load, preprocess, split, and create data loaders for model training. It 
leverages ZenML's pipeline and step functionality for structured data processing.

Classes:
    TrainingPipelineParams: Configuration parameters for controlling the pipeline steps, 
                            including batch size, tokenizer, and split ratio.

Functions:
    load: A ZenML step to load the raw tweet data.
    preprocess: A ZenML step to preprocess the tweet data by cleaning and standardizing text.
    split: A ZenML step to split the dataset into training and validation sets.
    prepare_dataloaders: A ZenML step to prepare PyTorch-compatible data loaders for the model.
    training_data_pipeline: The complete ZenML pipeline that chains together the steps.

Usage:
    Run this script directly to initialize and execute the pipeline with specified parameters.
"""

import os
import sys

import pandas as pd
from zenml.pipelines import pipeline
from zenml.steps import step, BaseParameters

from config import config
from src import data


class TrainingPipelineParams(BaseParameters):
    """Configuration parameters for the training data pipeline.

    Attributes:
        batch_size (int): The number of samples per batch for data loaders.
        tokenizer_name (str): The name of the tokenizer to use for tokenizing text data.
        split_ratio (float): The ratio for splitting the data into training and validation sets.
    """
    batch_size: int
    tokenizer_name: str
    split_ratio: float


@step(enable_cache=False)
def load() -> pd.DataFrame:
    """Loads the raw tweet data as a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded raw tweet data.
    """
    return data.load_data()


@step(enable_cache=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses tweet data by removing URLs and setting source flags.

    Args:
        df (pd.DataFrame): The raw tweet data.

    Returns:
        pd.DataFrame: The preprocessed tweet data.
    """
    return data.preprocess_data(df)


@step(enable_cache=False)
def split(params: TrainingPipelineParams, df: pd.DataFrame) -> dict:
    """Splits the preprocessed data into training and validation sets.

    Args:
        params (TrainingPipelineParams): The pipeline parameters, including split ratio.
        df (pd.DataFrame): The preprocessed tweet data.

    Returns:
        dict: A dictionary with 'train' and 'validation' datasets as DataFrames.
    """
    return data.split(df, ratio=params.split_ratio)


@step(enable_cache=False)
def prepare_dataloaders(params: TrainingPipelineParams,
                        train_test: dict) -> dict:
    """Prepares data loaders for the training and validation sets.

    Args:
        params (TrainingPipelineParams): Pipeline parameters specifying batch size and tokenizer.
        train_test (dict): A dictionary with 'train' and 'test' datasets as DataFrames.

    Returns:
        dict: A dictionary containing PyTorch data loaders for the training and validation sets.
            {
                'train': DataLoader for training set,
                'validation': DataLoader for validation set
            }
    """
    train = train_test['train']
    test = train_test['test']
    tokenizer = data.get_tokenizer(params.tokenizer_name)
    train_loader = data.get_loader(train,
                                   batch_size=params.batch_size,
                                   is_validation=False,
                                   tokenizer=tokenizer)
    val_loader = data.get_loader(test,
                                 batch_size=params.batch_size,
                                 is_validation=True,
                                 tokenizer=tokenizer)

    return {'train': train_loader, 'validation': val_loader}


@pipeline
def training_data_pipeline(load, preprocess, split, prepare_dataloaders):
    """Defines the ZenML pipeline for data loading, preprocessing, splitting, and preparing data loaders.

    Args:
        load (function): The step function to load raw data.
        preprocess (function): The step function to preprocess raw data.
        split (function): The step function to split preprocessed data.
        prepare_dataloaders (function): The step function to prepare data loaders for model training.

    This pipeline orchestrates each step in sequence, preparing data for model training.
    """
    tweets = load()
    preprocessed = preprocess(tweets)
    split_tweets = split(preprocessed)
    prepare_dataloaders(split_tweets)


# EXAMPLE USAGE
if __name__ == "__main__":
    pipeline_params = TrainingPipelineParams(
        batch_size=config.batch_size,
        tokenizer_name=config.tokenizer_name,
        split_ratio=config.split_ratio)

    load_instance = load()
    preprocess_instance = preprocess()
    split_instance = split(params=pipeline_params)
    prepare_dataloaders_instance = prepare_dataloaders(params=pipeline_params)

    training_data_pipeline_instance = training_data_pipeline(
        load=load_instance,
        preprocess=preprocess_instance,
        split=split_instance,
        prepare_dataloaders=prepare_dataloaders_instance)

    training_data_pipeline_instance.run()
