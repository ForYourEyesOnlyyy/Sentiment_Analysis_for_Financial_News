import os
import sys

import pandas as pd
from zenml.pipelines import pipeline
from zenml.steps import step, BaseParameters

# my modules imports
from utils import get_project_root

# Adding config path and importing config and src
config_path = os.path.join(get_project_root(), 'config')
src_path = os.path.join(get_project_root(), 'src')
sys.path.append(config_path)
sys.path.append(src_path)

import config
import data


class TrainingPipelineParams(BaseParameters):
    batch_size: int
    tokenizer_name: str
    split_ratio: float


@step
def load() -> pd.DataFrame:
    return data.load_data()


@step
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return data.preprocess_data(df)


@step
def split(params: TrainingPipelineParams, df: pd.DataFrame) -> dict:
    return data.split(df, ratio=params.split_ratio)


@step
def prepare_dataloaders(params: TrainingPipelineParams,
                        train_test: dict) -> dict:
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
