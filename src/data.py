import re
import pandas as pd

from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

data_path = "../data/processed/twitter-financial-news-sentiment/samples/sample1.csv"
label_column = "label"
text_column = "text"
has_source_column = "has_source"

def load_data(path):
    return pd.read_csv(path)


def preprocess(data):

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

    def __init__(self, texts, has_source, labels, tokenizer, max_length=32):
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
                                  return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'has_source': torch.tensor(has_source, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def split_and_get_loaders(data, ratio=0.33, batch_size=32, tokenizer='bert'):
    X = data.drop(columns=[label_column])
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=ratio,
                                                        random_state=42)

    if tokenizer == "bert":
        tok = AutoTokenizer.from_pretrained('bert-base-uncased')

        train_dataset = FinancialTweetsDataset(X_train[text_column].tolist(),
                                               X_train[has_source_column].tolist(),
                                               y_train.tolist(), tok)
        val_dataset = FinancialTweetsDataset(X_test[text_column].tolist(),
                                             X_test[has_source_column].tolist(),
                                             y_test.tolist(), tok)

        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
        return train_dataloader, val_dataloader
