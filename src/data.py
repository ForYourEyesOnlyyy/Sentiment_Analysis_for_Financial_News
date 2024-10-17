import sys
sys.path.append('/Users/maxmartyshov/Desktop/IU/year3/PMDL/Sentiment_Analysis_for_Financial_News/config')
import config
import re
import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('/Users/maxmartyshov/Desktop/IU/year3/PMDL/Sentiment_Analysis_for_Financial_News/config')

data_path = config.data_path
label_column = config.label_column
text_column = config.text_column
has_source_column = config.has_source_column

sentiments = config.sentiments


def make_dataframe_with_dummy_label(tweet: str) -> pd.DataFrame:
    return pd.DataFrame({text_column: [tweet], label_column: [0]})


def get_tokenizer(tokenizer_name: str = 'bert-base-uncased') -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_data() -> pd.DataFrame:
    return pd.read_csv(data_path)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:

    def process_source_links(row):
        if 'https' in row[text_column]:
            row[text_column] = re.sub(r'http\S+', '', row[text_column]).strip()
            row[has_source_column] = 1
        else:
            row[has_source_column] = 0
        return row

    data = data.apply(process_source_links, axis=1)
    return data


class FinancialTweetsDataset(Dataset):

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
        return len(self.texts)

    def __getitem__(self, idx):
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


def split(data, ratio=0.33):
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


def get_loader(data,
               batch_size=32,
               is_validation=False,
               tokenizer=any):
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
