import torch
import torch.nn as nn
from pytorch_transformers import *
import pandas as pd
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from tqdm import tqdm
import numpy as np
import random
import argparse
from utils import Params, RunningAverage, F1Avarage

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='data/trnTweet.txt', help="Directory containing the train dataset")
parser.add_argument('--test_dir', default='data/testTweet.txt', help="Directory containing the test dataset")
parser.add_argument('--save_dir', default='models/', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--batch_size', type=int, default=128, help="random seed for initialization")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--epoch_num', type=int, default=10, help="random seed for initialization")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--gpu', default=False, action='store_true', help="Whether to use GPUs if available")


def train(model, dataloader, optimizer, scheduler, params):
    best_val_f1 = 0.0
    for i in range(params.epoch_num):
        loss_avg = RunningAverage()
        train_data = tqdm(dataloader.data_iterator(_type='train',
                                                   batch_size=params.batch_size,
                                                   tokenizer=tokenizer,
                                                   total=(dataloader.train_size // params.batch_size)))
        for data, labels in train_data:
            batch_masks = data.gt(0)
            loss, logits = model(data, attention_mask=batch_masks, labels=labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # update the average loss
            loss_avg.update(loss.item())
            t.set_postfix(type='TRAIN',epoch=i,loss='{:05.3f}'.format(loss_avg()))

        metrics = validate(model, dataloader, params)
        if metrics['f1'] > best_val_f1:
            utils.save_checkpoint(({'epoch': epoch + 1,
                                    'state_dict': model_to_save.state_dict(),
                                    'optim_dict': optimizer_to_save.state_dict()},
                                    is_best=True,
                                    checkpoint=params.save_dir))


def validate(model, dataloader, params):
    model.eval()
    val_data = tqdm(dataloader.data_iterator(_type='val',
                                               batch_size=params.batch_size,
                                               tokenizer=tokenizer,
                                               total=(dataloader.val_size // params.batch_size)))
    f1 = F1Avarage()
    loss_avg = RunningAverage()
    with torch.no_grad():
        for data, labels in val_data:
            batch_masks = data.gt(0)
            loss, logits = model(data, attention_mask=batch_masks, labels=labels)
            predicted = logits.max(2)[1]
            f1.batch_update(pred=predicted, true=labels)
            loss_avg.update(loss.item())

    metrics = {}
    metrics['loss'] = loss_avg()
    metrics['f1'] = f1()
    return metrics




if __name__ == '__main__':
    args = parser.parse_args()

    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    params.save_dir = args.save_dir
    params.batch_size = args.batch_size
    params.epoch_num = args.epoch_num

    dataloader = DataLoader(path_to_train=args.train_dir, path_to_test=args.test_dir, seed=params.seed, device=params.device)

    # Training
    params.lr = 1e-3
    params.max_grad_norm = 1.0
    params.num_total_steps = (dataloader.train_size // params.batch_size)*params.epoch_niter
    params.num_warmup_steps = 1000

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device=params.device)

    optimizer = AdamW(model.parameters(), lr=params.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=params.num_warmup_steps, t_total=params.num_total_steps)  # PyTorch scheduler

    train(model, dataloader, optimizer, scheduler, params)
