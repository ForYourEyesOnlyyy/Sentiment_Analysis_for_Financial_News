from zenml.steps import step
from zenml.pipelines import pipeline

import pandas as pd

import sys 
sys.path.append("/Users/maxmartyshov/Desktop/IU/year3/PMDL/Sentiment_Analysis_for_Financial_News/src")

import data


@step
def load() -> pd.DataFrame:
    return data.load_data()


@step
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return data.preprocess_data(df)


@step
def split(df: pd.DataFrame, ratio: float = 0.33) -> dict:
    return data.split(df, ratio=ratio)


@step
def prepare_dataloaders(train_test: dict,
                        batch_size: int = 32,
                        tokenizer: str = 'bert-base-uncased') -> dict:
    train = train_test['train']
    test = train_test['test']

    train_loader = data.get_loader(train,
                                   batch_size=batch_size,
                                   is_validation=False,
                                   tokenizer_name=tokenizer)
    val_loader = data.get_loader(test,
                                 batch_size=batch_size,
                                 is_validation=True,
                                 tokenizer_name=tokenizer)

    return {'train': train_loader, 'validation': val_loader}


@pipeline
def training_data_pipeline(load, preprocess, split, prepare_dataloaders):
    tweets = load()
    preprocessed = preprocess(tweets)
    split_tweets = split(preprocessed)
    prepare_dataloaders(split_tweets)


training_data_pipeline_instance = training_data_pipeline(
    load=load(),
    preprocess=preprocess(),
    split=split(),
    prepare_dataloaders=prepare_dataloaders())

training_data_pipeline_instance.run()
