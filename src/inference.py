"""
inference.py

This module provides functions for loading models and performing inference on financial tweets, 
returning sentiment predictions. It includes functions for loading a model from the MLflow registry, 
running inference, and generating sentiment predictions using tokenized data.

Functions:
    load_model_from_registry: Loads a specified version of a model from the MLflow model registry.
    predict: Generates a sentiment prediction for a given dataset sample.
    run_inference: Runs the entire inference pipeline on a single tweet.
    predict_sentiment: A high-level function to predict sentiment for a given tweet.

Usage:
    This module is designed for use in other modules to perform inference on tweets. 
    It requires a model registered in the MLflow registry and a compatible tokenizer.
"""

import os
import sys
from time import time

import mlflow
from mlflow.tracking import MlflowClient
import torch

from src import data
from config import config

client = MlflowClient()


def load_model_from_registry(model_name: str, version="champion") -> any:
    """Loads a model from the MLflow registry by name and version.

    Args:
        model_name (str): The name of the model in the MLflow registry.
        version (str, optional): The version of the model to load. 
            Defaults to "champion" (latest best-performing model).

    Returns:
        any: The loaded PyTorch model.

    Raises:
        ValueError: If the specified version does not exist in the registry.
    """
    if version == "champion":
        global client
        champion_version = client.get_model_version_by_alias(
            name=model_name, alias="champion").version
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}/{champion_version}")
    else:
        model = mlflow.pytorch.load_model(
            model_uri=f"models:/{model_name}/{version}")

    print(f"Model {model_name} loaded successfully")
    return model


def predict(model: any, dataset: data.FinancialTweetsDataset,
            device: str) -> str:
    """Generates a sentiment prediction for a single tweet sample.

    Args:
        model (any): The pre-trained PyTorch model used for predictions.
        dataset (FinancialTweetsDataset): The dataset containing tokenized input for the tweet.
        device (str): The device on which the model will perform inference (e.g., 'cpu' or 'cuda').

    Returns:
        str: The predicted sentiment label.

    Raises:
        ValueError: If the model returns more than one prediction for a single sample.
    """
    dataset = dataset[0]
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_ids = dataset['input_ids'].unsqueeze(0).to(
            device)  # Adding batch dimension
        attention_mask = dataset['attention_mask'].unsqueeze(0).to(
            device)  # Adding batch dimension
        has_source = dataset['has_source'].unsqueeze(0).to(
            device)  # Adding batch dimension
        logits = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       has_source=has_source)
        _, preds = torch.max(logits, dim=1)

        preds = preds.cpu().numpy()

        if len(preds) > 1:
            raise ValueError("Model returned more than one prediction")
        else:
            return data.sentiments.get(int(preds[0]), 'Unknown')


def run_inference(tweet: str, tokenizer: any, max_length: int, model: any,
                  device: str) -> str:
    """Processes a tweet and runs inference to obtain a sentiment prediction.

    Args:
        tweet (str): The tweet text to analyze.
        tokenizer (any): Tokenizer used to preprocess the tweet.
        max_length (int): Maximum token length for the input sequence.
        model (any): The pre-trained model to use for prediction.
        device (str): The device on which the model will run (e.g., 'cpu' or 'cuda').

    Returns:
        str: The predicted sentiment label.
    """
    tweet_df = data.make_dataframe_with_dummy_label(tweet)
    tweet_df = data.preprocess_data(tweet_df, balance=False)
    tweet_dataset = data.FinancialTweetsDataset(
        tweet_df[data.text_column], tweet_df[data.has_source_column],
        tweet_df[data.label_column], tokenizer, max_length)
    return predict(model, tweet_dataset, device)


def predict_sentiment(tweet: str, tokenizer: any, model: any) -> str:
    """A high-level function to predict the sentiment of a tweet.

    Args:
        tweet (str): The tweet text for sentiment prediction.
        tokenizer (any): The tokenizer to preprocess the text input.
        model (any): The pre-trained model used for sentiment analysis.

    Returns:
        str: The predicted sentiment label.
    """
    max_length = config.max_length
    device = config.device
    return run_inference(tweet, tokenizer, max_length, model, device)


# EXAMPLE USAGE
if __name__ == "__main__":
    model = load_model_from_registry('simple_sentiment_analysis_model')
    tokenizer = data.get_tokenizer('bert-base-uncased')
    start = time()
    test_tweet = "$ANCUF: BMO Capital Markets ups to Outperform"
    print(predict_sentiment(test_tweet, tokenizer, model))
    print('Inference time:', time() - start)
