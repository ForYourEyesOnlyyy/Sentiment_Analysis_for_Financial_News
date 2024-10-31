import torch
import torch.nn as nn
from transformers import BertModel

class EnhancedSentimentAnalysisModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=3):
        super(EnhancedSentimentAnalysisModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size + 1, 128)
        self.activation = nn.ReLU()
        
        self.dropout = nn.Dropout(0.5)

        self.output_layer = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask, has_source):
        embeddings = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output

        has_source = has_source.unsqueeze(1)
        combined_input = torch.cat((embeddings, has_source), dim=1)

        hidden_out = self.activation(self.hidden_layer(combined_input))
        regularized_out = self.dropout(hidden_out)

        logits = self.output_layer(regularized_out)

        return logits
