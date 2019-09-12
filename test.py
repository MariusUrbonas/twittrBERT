import torch
import torch.nn as nn
from pytorch_transformers import *
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from tqdm import tqdm
import numpy as np
import argparse
from dataloader import DataLoader
from utils import Params, RunningAverage, Metrics, Stats, save_checkpoint, load_checkpoint
from models import DistilBertForTokenClassification

parser = argparse.ArgumentParser()
parser.add_argument('--test_data', default='data/testTweet.txt', help="Directory containing the test dataset")
parser.add_argument('--restore_file', default='models/', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--tag', default='experiment_0', help="Tag for experiment")
parser.add_argument('--gpu', default=False, action='store_true', help="Whether to use GPUs if available")
parser.add_argument('--batch_size', type=int, default=128, help="random seed for initialization")
parser.add_argument('--distil', default=False, action='store_true', help="Use Distiled Bert Model")

def test(model, dataloader, params):
    val_data = tqdm(dataloader.data_iterator(data_type='test',
                                             batch_size=params.batch_size),
                                             total=(dataloader.size()[0] // params.batch_size))
    metrics = Metrics()
    loss_avg = RunningAverage()
    with torch.no_grad():
        for data, labels in val_data:
            model.eval()
            data = torch.tensor(data, dtype=torch.long).to(params.device)
            labels = torch.tensor(labels, dtype=torch.long).to(params.device)

            batch_masks = data != 0

            loss, logits = model(data, attention_mask=batch_masks, labels=labels)

            predicted = logits.max(2)[1]
            metrics.update(batch_pred=predicted.cpu().numpy(), batch_true=labels.cpu().numpy(), batch_mask=batch_masks.cpu().numpy())
            loss_avg.update(torch.mean(loss).item())
            val_data.set_postfix(type='VAL',loss='{:05.3f}'.format(loss_avg()))
    metrics.loss = loss_avg()
    return metrics


if __name__ == '__main__':
    args = parser.parse_args()
    params = Params()

    params.device = torch.device('cuda' if args.gpu else 'cpu')

    params.tag = args.tag
    params.batch_size = args.batch_size

    test_dataloader = DataLoader(path_to_data=args.test_data, is_train=False)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataloader.pre_encode(tokenizer)

    model = DistilBertForTokenClassification(2, args.top_rnn) if args.distil else BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(params.device)

    checkp = load_checkpoint(args.restore_file)
    model.load_state_dict(checkp['state_dict'])

    metrics = test(model, test_dataloader, params)
    print('On test set: F1={}, Loss={}'.format(metrics.f1(), metrics.loss))
