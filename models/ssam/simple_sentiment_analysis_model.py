import torch.nn as nn
import torch

from transformers import BertModel


class SentimentAnalysisModel(nn.Module):

    def __init__(self, bert_model_name='bert-base-uncased', num_labels=3):
        super(SentimentAnalysisModel, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        self.linear1 = nn.Linear(self.bert.config.hidden_size + 1, num_labels)

        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, has_source):
        embeddings = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask).pooler_output
        has_source = has_source.unsqueeze(1)
        combined_input = torch.cat((embeddings, has_source), dim=1)

        regularized = self.dropout(combined_input)
        logits = self.linear1(regularized)

        return logits
