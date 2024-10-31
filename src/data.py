"""
data.py

This module contains functions and classes for handling and preprocessing tweet data related to financial sentiment analysis. 
It includes functions to load data, preprocess tweet text, split the data into training and testing sets, 
and create PyTorch-compatible DataLoader objects.

Classes:
    FinancialTweetsDataset: A PyTorch Dataset for financial tweets, designed to work with a DataLoader.

Functions:
    make_dataframe_with_dummy_label: Creates a single-row DataFrame with a dummy label for a given tweet.
    get_tokenizer: Loads a pre-trained tokenizer for text processing.
    load_data: Loads tweet data from a CSV file specified in the configuration.
    preprocess_data: Cleans tweet text and flags the presence of source links.
    split: Splits the dataset into training and testing sets.
    get_loader: Creates a DataLoader for batching and iterating over the dataset.

Usage:
    This module is primarily used to prepare tweet data for sentiment analysis modeling, 
    including tokenization, batching, and feature extraction.

"""

import re
import warnings

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from config import config

warnings.filterwarnings("ignore", category=FutureWarning)

data_path = config.data_path
label_column = config.label_column
text_column = config.text_column
has_source_column = config.has_source_column

sentiments = config.sentiments


def make_dataframe_with_dummy_label(tweet: str) -> pd.DataFrame:
    """Creates a DataFrame containing a single tweet with a dummy label.

    Args:
        tweet (str): The tweet text to include in the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with a single row containing the tweet text
            and a dummy label.
    """
    return pd.DataFrame({text_column: [tweet], label_column: [0]})


def get_tokenizer(tokenizer_name: str = 'bert-base-uncased') -> AutoTokenizer:
    """Retrieves a pre-trained tokenizer.

    Args:
        tokenizer_name (str): The name of the tokenizer model to use. Defaults to 'bert-base-uncased'.

    Returns:
        AutoTokenizer: A pre-trained tokenizer object from Hugging Face Transformers.
    """
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_data() -> pd.DataFrame:
    """Loads data from a CSV file specified in the config.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} tweets from {data_path}")
    return df


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses data by cleaning text, removing URLs, setting a source flag, and balancing class distributions.

    Args:
        data (pd.DataFrame): Raw tweet data to preprocess.

    Returns:
        pd.DataFrame: The preprocessed data with:
            - Cleaned tweet text (extra spaces and punctuation removed).
            - URLs removed from text, with an added column indicating the presence of a source link.
            - Balanced class distributions by downsampling the majority class to match minority classes.

    Steps:
        1. Clean text by removing unnecessary punctuation and spaces.
        2. Remove URLs from tweet text and add a binary flag (`has_source`) indicating if a URL was present.
        3. Balance the dataset to handle class imbalance by downsampling the majority class.
    """

    def clean_text(text):
        text = re.sub(r'\s+,', ',', text)  # Remove spaces before commas
        text = re.sub(r'[\'".]+$', '', text)  # Remove quotes and full stops at the end
        return text


    def process_source_links(row):
        if 'https' in row[text_column]:
            row[text_column] = re.sub(r'http\S+', '', row[text_column]).strip()
            row[has_source_column] = 1
        else:
            row[has_source_column] = 0
        return row
    
    def balance_dataset(df):
        # Separate majority and minority classes
        df_majority = df[df.label == 2]
        df_minority_1 = df[df.label == 1]
        df_minority_0 = df[df.label == 0]

        # Downsample majority class
        df_majority_downsampled = resample(df_majority, 
                                        replace=False,    # sample without replacement
                                        n_samples=len(df_minority_1),  # to match minority class
                                        random_state=42)  # reproducible results

        # Combine minority class with downsampled majority class
        df_balanced = pd.concat([df_majority_downsampled, df_minority_1, df_minority_0])

        return df_balanced

    data = data.apply(process_source_links, axis=1)
    data[text_column] = data[text_column].apply(clean_text)
    data = balance_dataset(data)
    return data


class FinancialTweetsDataset(Dataset):
    """A custom dataset class for financial tweets, compatible with PyTorch DataLoader.

    Args:
        texts (list): List of tweet texts.
        has_source (list): List indicating the presence of source links.
        labels (list): List of sentiment labels for each tweet.
        tokenizer (AutoTokenizer, optional): Pre-trained tokenizer for text tokenization.
        max_length (int, optional): Maximum length for tokenized text. Defaults to 100.

    Attributes:
        texts (list): List of tweet texts.
        has_source (list): List indicating the presence of source links.
        labels (list): List of sentiment labels for each tweet.
        tokenizer (AutoTokenizer): Pre-trained tokenizer for text tokenization.
        max_length (int): Maximum length for tokenized text.
    """

    def __init__(self,
                 texts,
                 has_source,
                 labels,
                 tokenizer=get_tokenizer(),
                 max_length=100):
        self.texts = texts
        self.has_source = has_source
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """Retrieves a single data sample at a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing input IDs, attention mask, source presence flag, and label.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        has_source = self.has_source[idx]

        encoding = self.tokenizer(text,
                                  padding="max_length",
                                  max_length=self.max_length,
                                  truncation=True,
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'has_source': torch.tensor(has_source, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def split(data: pd.DataFrame, ratio: float = 0.33) -> dict:
    """Splits the data into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to split.
        ratio (float, optional): The ratio of the test set size to the total dataset size. Defaults to 0.33.

    Returns:
        dict: A dictionary with 'train' and 'test' datasets, each containing text, source presence, and labels.
    """
    X = data.drop(columns=[label_column])
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=ratio,
                                                        random_state=42)
    return {
        "train": {
            text_column: X_train[text_column],
            has_source_column: X_train[has_source_column],
            label_column: y_train
        },
        "test": {
            text_column: X_test[text_column],
            has_source_column: X_test[has_source_column],
            label_column: y_test
        }
    }


def get_loader(
    data: dict,
    batch_size: int = 32,
    is_validation: bool = False,
    tokenizer: AutoTokenizer = get_tokenizer()
) -> DataLoader:
    """Creates a DataLoader for the given dataset.

    Args:
        data (dict): Dictionary containing 'text', 'has_source', and 'label' data.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        is_validation (bool, optional): Whether the loader is for validation data (disables shuffling). Defaults to False.
        tokenizer (AutoTokenizer, optional): Tokenizer to use for text processing.

    Returns:
        DataLoader: A DataLoader object for batching and iterating over the dataset.
    """
    texts = data[text_column].tolist()
    has_source = data[has_source_column].tolist()
    labels = data[label_column].tolist()

    dataset = FinancialTweetsDataset(texts,
                                     has_source,
                                     labels,
                                     tokenizer=tokenizer)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=(not is_validation))
    return dataloader
