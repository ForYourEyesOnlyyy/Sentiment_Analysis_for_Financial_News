"""
enhanced_sentiment_analysis_model.py

This module defines an enhanced BERT-based model architecture for sentiment analysis, 
specifically designed to incorporate additional features alongside textual embeddings. 
The model combines BERT-generated embeddings with an extra feature indicating the presence 
of a source link in the input, allowing it to account for contextual relevance in the sentiment prediction. 
This combined input is processed through a hidden layer and dropout regularization to output sentiment logits.

Classes:
    EnhancedSentimentAnalysisModel: A sentiment analysis model that extends BERT with an 
                                    additional input feature for more nuanced classification.

Usage:
    The model is suitable for sentiment classification tasks involving financial or other specialized text data. 
    Initialize with the desired parameters and provide input tensors (`input_ids`, `attention_mask`, 
    and `has_source`) for forward passes.
"""

import torch
import torch.nn as nn
from transformers import BertModel


class EnhancedSentimentAnalysisModel(nn.Module):
    """Enhanced BERT-based sentiment analysis model with an additional input feature.

    This model utilizes BERT embeddings combined with an extra feature input, processed 
    through a fully connected hidden layer, to enhance sentiment classification performance. 
    Dropout regularization and a non-linear activation function are applied to reduce overfitting 
    and improve generalization.

    Args:
        bert_model_name (str): The name of the pre-trained BERT model to use. Defaults to 'bert-base-uncased'.
        num_labels (int): The number of sentiment classes. Defaults to 3.

    Attributes:
        bert (BertModel): Pre-trained BERT model for generating embeddings from text.
        hidden_layer (nn.Linear): Linear layer to process the combined BERT embeddings and source feature.
        activation (nn.ReLU): ReLU activation function applied to hidden layer output.
        dropout (nn.Dropout): Dropout layer for regularization, with a dropout probability of 0.5.
        output_layer (nn.Linear): Final linear layer to produce sentiment classification logits.
    """

    def __init__(self, bert_model_name='bert-base-uncased', num_labels=3):
        super(EnhancedSentimentAnalysisModel, self).__init__()

        # Initialize BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Hidden layer to process BERT embeddings with additional input
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size + 1, 128)
        self.activation = nn.ReLU()

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Output layer to generate sentiment logits
        self.output_layer = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask, has_source):
        """Performs a forward pass to compute sentiment logits.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs for each word in the input text.
            attention_mask (torch.Tensor): Tensor indicating non-padding tokens in `input_ids`.
            has_source (torch.Tensor): Tensor indicating the presence (1) or absence (0) of a source link in the text.

        Returns:
            torch.Tensor: Logits representing the model's sentiment predictions for each input sample.
        """
        # Generate embeddings using BERT
        embeddings = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask).pooler_output

        # Add source feature and concatenate with embeddings
        has_source = has_source.unsqueeze(1)
        combined_input = torch.cat((embeddings, has_source), dim=1)

        # Apply hidden layer, activation, and dropout
        hidden_out = self.activation(self.hidden_layer(combined_input))
        regularized_out = self.dropout(hidden_out)

        # Output sentiment logits
        logits = self.output_layer(regularized_out)

        return logits
