import os
import sys
from time import time

import mlflow
from mlflow.tracking import MlflowClient
import torch

from src import data
from src.src_utils import get_project_root

# Adding config path and importing config
config_path = os.path.join(get_project_root(), 'config')
sys.path.append(config_path)
from config import config

client = MlflowClient()


def load_model_from_registry(model_name, version="champion"):
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
    tweet_df = data.make_dataframe_with_dummy_label(tweet)
    tweet_df = data.preprocess_data(tweet_df)
    tweet_dataset = data.FinancialTweetsDataset(
        tweet_df[data.text_column], tweet_df[data.has_source_column],
        tweet_df[data.label_column], tokenizer, max_length)
    return predict(model, tweet_dataset, device)


def predict_sentiment(tweet: str, tokenizer: any, model: any) -> str:
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
