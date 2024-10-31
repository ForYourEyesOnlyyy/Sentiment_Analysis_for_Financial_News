import torch
import torch.nn as nn
from transformers import BertModel

class LSTMSentimentAnalysisModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_labels=3, hidden_dim=128):
        super(LSTMSentimentAnalysisModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-3:].parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(self.bert.config.hidden_size, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_labels)
        
        self.dropout = nn.Dropout(0.5)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim * 2)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, input_ids, attention_mask, has_source):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(embeddings)
        
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(lstm_out * attention_weights, dim=1)

        out = self.batchnorm1(attended_output)
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.batchnorm2(out)
        logits = self.fc2(out)
        
        return logits
