"""
bert_lstm_sentiment_analysis_model.py

This module defines a BERT-LSTM-based model architecture for sentiment analysis on financial text data. 
The model leverages a pre-trained BERT model for embedding input text and uses an LSTM layer to capture 
contextual dependencies in the sequence. Additionally, an attention mechanism is applied over the LSTM 
outputs, followed by fully connected layers to produce sentiment classification logits.

Classes:
    LSTMSentimentAnalysisModel: A hybrid model combining BERT embeddings, LSTM sequence modeling, and 
                                attention for enhanced sentiment classification.

Usage:
    The model can be used for sentiment analysis tasks, especially with financial text data. 
    Initialize the model with the desired parameters and provide input tensors (`input_ids`, 
    `attention_mask`, and optionally `has_source`) to obtain sentiment logits.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class LSTMSentimentAnalysisModel(nn.Module):
    """BERT-LSTM-based model with attention for sentiment analysis.

    This model uses BERT for generating text embeddings and a bidirectional LSTM layer 
    for capturing sequential information. An attention mechanism is applied over LSTM outputs 
    to emphasize important tokens in the sequence, and the attended output is passed through 
    fully connected layers for sentiment classification.

    Args:
        bert_model_name (str): The name of the pre-trained BERT model to use. Defaults to 'bert-base-uncased'.
        num_labels (int): The number of sentiment classes. Defaults to 3.
        hidden_dim (int): The hidden dimension size for the LSTM layer. Defaults to 128.

    Attributes:
        bert (BertModel): Pre-trained BERT model for text embeddings.
        lstm (nn.LSTM): Bidirectional LSTM layer to process BERT embeddings.
        attention (nn.Linear): Linear layer to compute attention weights over LSTM outputs.
        fc1 (nn.Linear): First fully connected layer for reducing dimensionality.
        fc2 (nn.Linear): Second fully connected layer for outputting sentiment logits.
        dropout (nn.Dropout): Dropout layer for regularization.
        batchnorm1 (nn.BatchNorm1d): Batch normalization layer for attended LSTM output.
        batchnorm2 (nn.BatchNorm1d): Batch normalization layer for output of first fully connected layer.
    """

    def __init__(self, bert_model_name='bert-base-uncased', num_labels=3, hidden_dim=128):
        super(LSTMSentimentAnalysisModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT layers except for the last three for efficiency
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-3:].parameters():
            param.requires_grad = True

        # Define LSTM layer
        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=1, 
                            bidirectional=True, batch_first=True)
        
        # Attention layer to compute weights over LSTM output
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_labels)
        
        # Regularization layers
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim * 2)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, input_ids, attention_mask, has_source=None):
        """Performs a forward pass through the BERT-LSTM model to obtain sentiment logits.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs for each word in the input text.
            attention_mask (torch.Tensor): Tensor indicating non-padding tokens in `input_ids`.
            has_source (torch.Tensor, optional): Tensor indicating the presence (1) or absence (0) of a 
                                                 source link. Currently unused in this model.

        Returns:
            torch.Tensor: Logits representing the model's sentiment predictions for each input sample.
        """
        # Extract embeddings from BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        
        # LSTM layer for sequential processing
        lstm_out, _ = self.lstm(embeddings)
        
        # Compute attention weights over LSTM output and apply them
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)

        # Batch normalization, dropout, and fully connected layers for classification
        out = self.batchnorm1(attended_output)
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.batchnorm2(out)
        logits = self.fc2(out)
        
        return logits
