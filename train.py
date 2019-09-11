import torch
import torch.nn as nn
from pytorch_transformers import *
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from tqdm import tqdm
import numpy as np
import random
import argparse
from dataloader import DataLoader
from utils import Params, RunningAverage, Metrics, Stats, save_checkpoint, load_checkpoint
from models import DistilBertForTokenClassification

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', default='data/trnTweet.txt', help="Directory containing the train dataset")
parser.add_argument('--test_data', default='data/testTweet.txt', help="Directory containing the test dataset")
parser.add_argument('--save_dir', default='models/', help="Directory containing the BERT model in PyTorch")
parser.add_argument('--tag', default='experiment_0', help="Tag for experiment")
parser.add_argument('--batch_size', type=int, default=128, help="random seed for initialization")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--save_freq', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=10, help="random seed for initialization")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gpu', default=False, action='store_true', help="Whether to use GPUs if available")
parser.add_argument('--top_rnn', default=False, action='store_true', help="Use Rnn on  top if using custom Distil bert")
parser.add_argument('--data_parr', default=False, action='store_true', help="Use data parralel")
parser.add_argument('--distil', default=False, action='store_true', help="Use Distiled Bert Model")



def train(model, dataloader, optimizer, scheduler, params):
    print("Starting training...")
    best_val_f1 = 0.0
    #print(params.save_dir, params.tag)
    stats = Stats(params.save_dir, params.tag)
    for epoch in range(params.epoch_num):
        loss_avg = RunningAverage()
        train_data = tqdm(dataloader.data_iterator(data_type='train',
                                                    batch_size=params.batch_size),
                                                   total=(len(dataloader.train) // params.batch_size))
        optimizer.zero_grad()
        for data, labels in train_data:
            data = torch.tensor(data, dtype=torch.long).to(params.device)
            labels = torch.tensor(labels, dtype=torch.long).to(params.device)

            batch_masks = (data != 0)
            output = model(data, attention_mask=batch_masks, labels=labels)

            loss = torch.mean(output[0])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # update the average loss
            loss_avg.update(loss.item())
            train_data.set_postfix(type='TRAIN',epoch=epoch,loss='{:05.3f}'.format(loss_avg()))

        metrics = validate(model, dataloader, params)
        print('After {} epochs: F1={}, Loss={}'.format(epoch , metrics.f1(), metrics.loss))
        stats.update(metrics, epoch, loss_avg())
        stats.save()

        if metrics.f1() > best_val_f1:
            best_val_f1 = metrics.f1()
            save_checkpoint({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                    is_best=True,
                                    tag=params.tag,
                                    epoch=epoch,
                                    score=metrics.f1(),
                                    checkpoint=params.save_dir)

def validate(model, dataloader, params):
    model.eval()
    val_data = tqdm(dataloader.data_iterator(data_type='val',
                                               batch_size=params.batch_size),
                                               total=(len(dataloader.val) // params.batch_size))
    metrics = Metrics()
    loss_avg = RunningAverage()
    with torch.no_grad():
        for data, labels in val_data:
            data = torch.tensor(data, dtype=torch.long).to(params.device)
            labels = torch.tensor(labels, dtype=torch.long).to(params.device)

            batch_masks = data != 0

            loss, logits = model(data, attention_mask=batch_masks, labels=labels)
            print(">>>>> logits shape: ", logits.size())
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    params.tag = args.tag
    params.save_dir = args.save_dir
    params.batch_size = args.batch_size
    params.epoch_num = args.num_epoch
    params.save_freq = args.save_freq

    dataloader = DataLoader(path_to_train=args.train_data, path_to_test=args.test_data, seed=params.seed, shuffle=True)

    params.lr = args.lr
    params.max_grad_norm = 1.0
    params.num_total_steps = (len(dataloader.train) // params.batch_size) * params.epoch_num
    params.num_warmup_steps = params.num_total_steps // 100

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataloader.pretokenize(tokenizer)

    model = DistilBertForTokenClassification(2, args.top_rnn) if args.distil else BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(params.device)

    optimizer = AdamW(model.parameters(), lr=params.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=params.num_warmup_steps, t_total=params.num_total_steps)  # PyTorch scheduler

    if args.restore_file is not None:
        checkp = load_checkpoint(args.restore_file)
        model.load_state_dict(checkp['state_dict'])

    train(model, dataloader, optimizer, scheduler, params)

