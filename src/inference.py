import mlflow
from mlflow.tracking import MlflowClient

import torch

from time import time

import data

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


def predict(model_name: str, dataset: data.FinancialTweetsDataset,
            device: str) -> str:
    dataset = dataset[0]

    model = load_model_from_registry(model_name)
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


# EXAMPLE USAGE
'''
    test_tweet = "Copa Holdings stock price target raised to $130 from $103 at Deutsche Bank"
    dataframe = data.make_dataframe_with_dummy_label(test_tweet)
    dataframe = data.preprocess_data(dataframe)
    dataset = data.FinancialTweetsDataset(dataframe[data.text_column], dataframe[data.has_source_column], dataframe[data.label_column])
    dataset = dataset[0]
    pred = predict('simple_sentiment_analysis_model', dataset, 'mps')
    print(pred)
'''

# def run_inference(tweet: str, tokenizer: any, max_length: int, model_name: str, device: str)-> str:
#     tweet_df = data.make_dataframe_with_dummy_label(tweet)
#     tweet_df = data.preprocess_data(tweet_df)
#     tweet_dataset = data.FinancialTweetsDataset(tweet_df[data.text_column], tweet_df[data.has_source_column], tweet_df[data.label_column], tokenizer, max_length)
#     return predict(model_name, tweet_dataset, device)

# if __name__ == "__main__":
#     start = time()
#     test_tweet = "$ANCUF: BMO Capital Markets ups to Outperform"
#     tokenizer = data.get_tokenizer('bert-base-uncased')
#     max_length = 100
#     model_name = 'simple_sentiment_analysis_model'
#     device = 'cpu'
#     print(run_inference(test_tweet, tokenizer, max_length, model_name, device))
#     print('Inference time:', time() - start)
