"""
simple_sentiment_analysis_model.py

This module defines a simple BERT-based model architecture for sentiment analysis on financial tweets. 
The model uses a pre-trained BERT model for embedding the input text, with an additional input feature 
indicating the presence of a source link. The BERT embeddings and source flag are combined and passed 
through a fully connected layer to output sentiment logits.

Classes:
    SentimentAnalysisModel: A BERT-based sentiment analysis model with an additional input feature 
                            to capture the presence of source links.

Usage:
    This model is designed for use with financial tweet data to predict sentiment labels. 
    Instantiate the model and provide input tensors for `input_ids`, `attention_mask`, and `has_source` 
    during forward passes.
"""

import torch.nn as nn
import torch
from transformers import BertModel


class SentimentAnalysisModel(nn.Module):
    """BERT-based model for financial tweet sentiment analysis.

    This model combines BERT embeddings with an additional input feature that indicates 
    the presence of a source link in the tweet. The combined features are passed through 
    a linear layer to output sentiment logits for classification.

    Args:
        bert_model_name (str): The name of the pre-trained BERT model to use. Defaults to 'bert-base-uncased'.
        num_labels (int): The number of sentiment classes. Defaults to 3.

    Attributes:
        bert (BertModel): The BERT model for generating embeddings.
        linear1 (nn.Linear): A linear layer that combines BERT embeddings with the source flag feature.
        dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self, bert_model_name='bert-base-uncased', num_labels=3):
        super(SentimentAnalysisModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.linear1 = nn.Linear(self.bert.config.hidden_size + 1, num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, has_source):
        """Performs a forward pass to obtain sentiment logits.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs for each word in the tweet input.
            attention_mask (torch.Tensor): Tensor indicating non-padding tokens in `input_ids`.
            has_source (torch.Tensor): Tensor indicating the presence (1) or absence (0) of a source link in the tweet.

        Returns:
            torch.Tensor: Logits representing the model's sentiment predictions for each input sample.
        """
        embeddings = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask).pooler_output
        has_source = has_source.unsqueeze(1)
        combined_input = torch.cat((embeddings, has_source), dim=1)

        regularized = self.dropout(combined_input)
        logits = self.linear1(regularized)

        return logits
