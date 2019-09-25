import torch
import torch.nn as nn
from pytorch_transformers import *
from tqdm import tqdm
import numpy as np
import random
import argparse
from dataset import KeyphraseData
from utils import Params, RunningAverage, Metrics, Stats, save_checkpoint
#from models_new import BertForKeyprhaseExtraction
from torch.utils.data import DataLoader


def train(model, dataset, optimizer, scheduler, params):
    print("Starting training...")
    best_val_loss = 100
    #print(params.save_dir, params.tag)
    train_set, val_set = dataset
    stats = Stats(params.save_dir, params.tag)
    for epoch in range(params.epoch_num):
        loss_avg = RunningAverage()
        train_data = tqdm(DataLoader(train_set, batch_size=params.batch_size, collate_fn=KeyphraseData.collate_fn),
                                                   total=(len(train_set) // params.batch_size))
        optimizer.zero_grad()
        model.zero_grad()
        for data, labels, mask in train_data:
            model.train()
            data = data.to(params.device)
            labels = labels.to(params.device)
            mask = mask.to(params.device)

            output = model(data, attention_mask=mask, labels=labels)

            loss = torch.mean(output[0])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)

            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            # update the average loss
            loss_avg.update(loss.item())
            train_data.set_postfix(type='TRAIN',epoch=epoch,loss='{:05.3f}'.format(loss_avg()))


        metrics = validate(model, val_set, params)
        print('After {} epochs: F1={}, Loss={}'.format(epoch , metrics.f1(), metrics.loss))
        stats.update(metrics, epoch, loss_avg())
        stats.save()
        if epoch % params.save_freq == 0 and params.save_checkpoints:
            save_checkpoint({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                    is_best=False,
                                    tag=params.tag,
                                    epoch=epoch,
                                    score=metrics.f1(),
                                    checkpoint=params.save_dir)
        if metrics.loss < best_val_loss:
            best_val_loss = metrics.loss
            save_checkpoint({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'optim_dict': optimizer.state_dict()},
                                    is_best=True,
                                    tag=params.tag,
                                    epoch='generic',
                                    score='epic',
                                    checkpoint=params.save_dir)
        break


def validate(model, val_set, params):
    val_data = tqdm(DataLoader(val_set, batch_size=params.batch_size, collate_fn=KeyphraseData.collate_fn),
                                               total=(len(val_set) // params.batch_size))
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
            break
    metrics.loss = loss_avg()
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='data/train_Tweets.csv', help="Directory containing the train dataset")
    parser.add_argument('--save_dir', default='models/', help="Directory containing the BERT model in PyTorch")
    parser.add_argument('--tag', default='exp_0.0', help="Tag for experiment")
    parser.add_argument('--batch_size', type=int, default=64, help="random seed for initialization")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--warmup_steps', type=int, default=400, help="random seed for initialization")
    parser.add_argument('--save_freq', type=int, default=-1)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('-l', '--load_checkpoint', default=None)
    parser.add_argument('--lr', type=float, default=5e-5)
    #parser.add_argument('--top_rnn', default=False, action='store_true', help="Use Rnn on  top if using custom Distil bert")
    #parser.add_argument('--distil', default=False, action='store_true', help="Use Distiled Bert Model")

    args = parser.parse_args()
    params = Params()

    # Use cuda if available
    if torch.cuda.is_available():
        print(">>> Using Cuda")
        params.device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    else:
        print(">>> Using Cpu")
        params.device = torch.device('cpu')

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    if args.train_data[-3:] == 'csv':
        train_dataset, val_dataset = KeyphraseData.load_train_csv(args.train_data, filter_lang='en')
    elif args.train_data[-3:] == 'txt':
        train_dataset, val_dataset = KeyphraseData.load_train_txt(args.train_data)
    else:
        raise Exception('Unsupported dataset extension: .{}'.format(args.train_data[-3:]))

    # Set parameter values 
    params.seed = args.seed
    params.tag = args.tag
    params.save_dir = args.save_dir
    params.batch_size = args.batch_size
    params.epoch_num = args.num_epoch
    params.save_freq = args.save_freq
    params.load_checkpoint = args.load_checkpoint
    params.lr = args.lr
    params.max_grad_norm = 1.0
    params.num_total_steps = (len(train_dataset)// params.batch_size) * params.epoch_num
    params.num_warmup_steps = args.warmup_steps

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    KeyphraseData.set_tokenizer(tokenizer)

    if args.load_checkpoint is not None:
        model = model = BertForTokenClassification.from_pretrained(args.restore_file, num_labels=2)
    else:
        model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

    if torch.cuda.device_count() > 1:
        print(">>> Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    model.to(params.device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=params.num_warmup_steps, t_total=params.num_total_steps)  # PyTorch scheduler

    data = train_dataset, val_dataset
    train(model, data, optimizer, scheduler, params)

