import torch
import torch.nn as nn
from pytorch_transformers import *
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from tqdm import tqdm
import numpy as np

class DistilBertForTokenClassification(torch.nn.Module):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]
    """
    def __init__(self, num_labels,top_rnn=False, hidden_size=768 ,hidden_dropout_prob=0.1):
        super(DistilBertForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.top_rnn=top_rnn
        if self.top_rnn:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        #self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)



    def forward(self, input_ids, labels=None, attention_mask=None):
        #with torch.no_grad():
        #outputs = self.bert(input_ids, attention_mask=attention_mask
        if self.top_rnn:
            input_ids = self.rnn(input_ids)
        sequence_output = self.dropout(input_ids)
        logits = self.classifier(sequence_output)

        outputs = (logits,)# + outputs[2:]  # add hidden states and attention if they are here
        #print(logits)
        #print(logits.view(-1, self.num_labels))
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                #print(attention_mask.size())
                #print(logits.size())
                active_loss = attention_mask.view(-1) == 1
                #print(active_loss)
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                #print(active_logits)
                active_labels = labels.view(-1)[active_loss]
                #print(active_labels)
                loss = loss_fct(active_logits, active_labels)
                #print(loss)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
