import numpy as np
import torch
import random
from tqdm import tqdm
from pathlib import Path
from utils import TweetProcesser
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class KeyphraseData(Dataset):
    tokenizer = None

    def __init__(self, data_df):
        self.data = data_df
        self.labels ={'O'   :0, 
                      'KP-W':1}

    @classmethod
    def load_train_csv(cls, path_to_csv, val_ratio=0.1, filter_lang=None):
        data = pd.read_csv(path_to_csv)
        if filter_lang is not None:
            data = cls.filter_lang(data, filter_lang)
        train, val = train_test_split(data, test_size=val_ratio)
        return cls(data_df=train), cls(data_df=val)

    @classmethod
    def load_test_csv(cls, path_to_csv, filter_lang=None):
        data = pd.read_csv(path_to_csv)
        if filter_lang is not None:
            data = cls.filter_lang(data, filter_lang)
        return cls(data_df=data)

    @classmethod
    def load_train_txt(cls, path_to_file, val_ratio=0.1, filter_lang=None):
        text, keyphrase = cls.load(path_to_file)
        data = {'text': text, 'keyphrase': keyphrase}
        data = pd.DataFrame(data=data)
        train, val = train_test_split(data, test_size=val_ratio)
        return cls(data_df=train), cls(data_df=val)        

    @classmethod
    def load_test_txt(cls, path_to_file, filter_lang=None):
        text, keyphrase = cls.load(path_to_file)
        data = {'text': text, 'keyphrase': keyphrase}
        data = pd.DataFrame(data=data)
        return cls(data_df=data)

    @staticmethod
    def load(path):
        print("Loading --{}-- ....\n".format(path))
        keyphrases = []
        tweets = []

        with open(path) as infile:
            for line in infile:
                t, key = line.strip().split('\t')
                tweets.append(t)
                keyphrases.append(key)
                #print(tweets[-1])
        print("{} Data Loaded....\n".format(len(tweets)))
        return tweets, keyphrases

    @staticmethod
    def filter_lang(data, lang):
        assert 'lang' in data
        pre_size = len(data)
        data = data[data.lang == lang]
        post_size = len(data)
        print('Filtered {} items, data contains {} items'.format(pre_size-post_size, post_size))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data
        data_point = data.take([idx])
        text, kp = data_point.text.item(), data_point.keyphrase.item()
        
        tokens, label, mask = self.prepare(text, kp)
        sample = {'tokens': tokens, 'kp': kp, 'token_ids': self.tokenizer.convert_tokens_to_ids(tokens), 'label': label, 'mask': mask}
        return sample

    @classmethod
    def set_tokenizer(cls, tokenizer):
        cls.tokenizer = tokenizer

    def prepare(self, text, keyphrase):
        if self.tokenizer is None:
            raise Exception('Tokenizer not set, use : KeyphraseData.set_tokenizer')

        text = TweetProcesser.replace_url(text)
        tokens = self.tokenizer.tokenize(text)
        tokens =  ['[CLS]'] + tokens + ['[SEP]']
        heads = self.get_heads(tokens)

        kp_tokens = self.tokenizer.tokenize(keyphrase)
        label = np.array([self.labels['KP-W'] if token in kp_tokens else self.labels['O'] for token in tokens])
        # only care about predicting the label for head token of the word
        label = np.multiply(heads, label)      
        return tokens, label, heads

    def get_heads(self, tokens):
        return np.array([1 if token[:2] != '##' else 0 for token in tokens])

    @staticmethod
    def collate_fn(batch):
        max_len = 512
        batch_size = len(batch)

        np_data = np.zeros((batch_size, max_len))
        np_labels = np.zeros((batch_size, max_len))
        np_mask = np.zeros((batch_size, max_len))

        max_len_token_ids = 0
        for i, item in enumerate(batch):
            len_token_ids = len(item['token_ids'])

            if len_token_ids > max_len:
                np_labels[i, :] = item['label'][:max_len]
                np_mask[i, :] = item['mask'][:max_len]
                max_len_token_ids = max_len
            else:
                np_data[i,:len_token_ids] = item['token_ids']
                np_labels[i, :len_token_ids] = item['label']
                np_mask[i, :len_token_ids] = item['mask']
                max_len_token_ids = max(max_len_token_ids, len_token_ids)

        data = torch.from_numpy(np_data[:,:max_len_token_ids]).type(dtype=torch.long)
        labels = torch.from_numpy(np_labels[:,:max_len_token_ids]).type(dtype=torch.long)
        mask = torch.from_numpy(np_mask[:,:max_len_token_ids]).type(dtype=torch.long)
        return data, labels, mask








