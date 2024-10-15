import mlflow
from mlflow.tracking import MlflowClient

import torch

from transformers import AutoTokenizer

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

def predict(model_name, dataset: data.FinancialTweetsDataset, device) -> str:
    model = load_model_from_registry(model_name)
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_ids = dataset['input_ids'].unsqueeze(0).to(device)  # Adding batch dimension
        attention_mask = dataset['attention_mask'].unsqueeze(0).to(device)  # Adding batch dimension
        has_source = dataset['has_source'].unsqueeze(0).to(device)  # Adding batch dimension
        logits = model(input_ids=input_ids, attention_mask=attention_mask, has_source=has_source)
        _, preds = torch.max(logits, dim=1)

        preds = preds.cpu().numpy()

        if len(preds) > 1:
            raise ValueError("Model returned more than one prediction")
        else:
            return data.sentiments.get(int(preds[0]), 'Unknown')
        
# test_tweet = "Copa Holdings stock price target raised to $130 from $103 at Deutsche Bank"
# dataframe = data.make_dataframe_with_dummy_label(test_tweet)
# dataframe = data.preprocess_data(dataframe)
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# dataset = data.FinancialTweetsDataset(dataframe[data.text_column], dataframe[data.has_source_column], dataframe[data.label_column], tokenizer)
# dataset = dataset[0]
# pred = predict('simple_sentiment_analysis_model', dataset, 'mps')
# print(pred)
        


