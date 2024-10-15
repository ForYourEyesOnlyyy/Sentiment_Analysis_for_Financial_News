from zenml.steps import step, BaseParameters
from zenml.pipelines import pipeline

from transformers import AutoTokenizer

import pandas as pd

import sys

sys.path.append(
    "/Users/maxmartyshov/Desktop/IU/year3/PMDL/Sentiment_Analysis_for_Financial_News/src"
)

import data
import inference


class InferencePipelineParams(BaseParameters):
    tweet: str
    tokenizer_name: str = 'bert-base-uncased'
    max_length: int = 100
    model_name: str = 'simple_sentiment_analysis_model'
    device: str = 'cpu'


@step
def make_dataframe(params: InferencePipelineParams) -> pd.DataFrame:
    return data.make_dataframe_with_dummy_label(params.tweet)


@step
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return data.preprocess_data(df)


@step
def make_dataset(params: InferencePipelineParams,
                 df: pd.DataFrame) -> data.FinancialTweetsDataset:
    return data.FinancialTweetsDataset(
        df[data.text_column],
        df[data.has_source_column], df[data.label_column],
        data.get_tokenizer(params.tokenizer_name), params.max_length)


@step
def predict(params: InferencePipelineParams,
            dataset: data.FinancialTweetsDataset) -> str:
    prediction = inference.predict(params.model_name, dataset, params.device)
    return prediction


@pipeline
def inference_pipeline(make_dataframe, preprocess, make_dataset, predict):
    tweet = make_dataframe()
    preprocessed = preprocess(tweet)
    dataset = make_dataset(preprocessed)
    predictions = predict(dataset)
    return predictions


if __name__ == "__main__":
    pipeline_params = InferencePipelineParams(
        tweet="$ANCUF: BMO Capital Markets ups to Outperform",
        tokenizer=data.get_tokenizer('bert-base-uncased'),
        max_length=100,
        model_name='simple_sentiment_analysis_model',
        device='mps')

    make_dataframe_instance = make_dataframe(params=pipeline_params)
    preprocess_instance = preprocess()
    make_dataset_instance = make_dataset(params=pipeline_params)
    predict_instance = predict(params=pipeline_params)

    inference_pipeline_instance = inference_pipeline(
        make_dataframe=make_dataframe_instance,
        preprocess=preprocess_instance,
        make_dataset=make_dataset_instance,
        predict=predict_instance)
    from time import time
    start = time()
    inference_pipeline_instance.run()
    # Access the prediction result from ZenML
    from zenml.client import Client
    inf_start = time()
    client = Client()
    pipeline = client.get_pipeline("inference_pipeline")
    latest_run = pipeline.last_run
    predict_step_result = latest_run.steps["predict"].output.load()

    # Printing the prediction
    print("Prediction:", predict_step_result)
    print("Time taken to run the pipeline + loading:", time() - start)
    print("Time taken to load the prediction:", time() - inf_start)
'''
    Time taken to run the pipeline + loading: 1.7046940326690674
    Time taken to load the prediction: 0.05875706672668457
    -------------------------------------------------------
    Time taken to run the pipeline + loading: 3.325381278991699
    Time taken to load the prediction: 0.07668924331665039
'''
