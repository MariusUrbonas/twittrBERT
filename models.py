import torch
import torch.nn as nn
from pytorch_transformers import *
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from tqdm import tqdm
import numpy as np


class DistilBertForTokenClassification(torch.nn.Module):

    def __init__(self, num_labels,top_rnn=False, hidden_size=768 ,hidden_dropout_prob=0.1, no_grad_bert=False):
        super(DistilBertForTokenClassification, self).__init__()
        self.num_labels     = num_labels
        self.no_grad_bert   = no_grad_bert
        self.top_rnn        = top_rnn
        if self.top_rnn:
            self.rnn        = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.bert           = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout        = nn.Dropout(hidden_dropout_prob)
        self.classifier     = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, labels=None, attention_mask=None):
        if self.no_grad_bert:
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask)

        if self.top_rnn:
            outputs = self.rnn(outputs)
        head_outputs = outputs[0]
        sequence_output = self.dropout(head_outputs)
        logits = self.classifier(sequence_output)

        outputs = (logits,)# + outputs[2:]  # add hidden states and attention if they are here
        #print(logits)
        #print(logits.view(-1, self.num_labels))
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
  
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

