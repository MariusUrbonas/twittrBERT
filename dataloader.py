import numpy as np
import torch
import random
from tqdm import tqdm
from pathlib import Path
import pickle


class DataLoader():

    def __init__(self, path_to_data, is_train=True, val_ratio=0.1, seed=42, shuffle=False):
        self.cache_path = Path('./__cache__')
        self.cache_train = Path('./__cache__') / 'train.cache'
        self.cache_test = Path('./__cache__') / 'test.caches'
        self.seed = seed
        self.is_train = is_train
        if is_train:
            self.data, self.val = self.load(path=path_to_data, split_val=True, val_ratio=val_ratio, shuffle=shuffle)
        else:
            self.data = self.load(path=path_to_data, split_val=False)
        self.pre_encoded = False


    def load(self, path: str, split_val=False, val_ratio=None, shuffle=False):
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

        if split_val:
            return self.split_validation(data=list(zip(tweets, keyphrases)), val_ratio=val_ratio, shuffle=shuffle)

        return list(zip(tweets, keyphrases))


    def split_validation(self, data, val_ratio, shuffle):
        size = len(data)
        split = int(size*(1-val_ratio))
        if shuffle:
            self.order = list(range(size))
            random.seed(self.seed)
            random.shuffle(self.order)
            data = [data[i] for i in self.order]
        #return train_list[:split],train_list[split:]
        return data[:split], data[split:]


    def pre_encode(self, encoder):
        # Load if cache exists
        if self.cache_path.exists():
            if self.is_train:
                if self.cache_train.exists():
                    train_dic = pickle.load(open(str(self.cache_train),'rb'))
                    self.train = np.ndarray(train_dic['train'])
                    self.val = np.ndarray(train_dic['val'])
                    self.pre_encoded = True
                    return
            else:
                if self.cache_test.exists():
                    test_dic = pickle.load(open(str(self.cache_test),'rb'))
                    self.test = np.ndarray(test_dic['test'])
                    self.pre_encoded = True
                    return


        print("Preencoding datasets")
        if self.is_train:
            to_encode = [('train', self.data),('val', self.val)]
        else:
            to_encode = [('test', self.data)]

        for data_type, data in to_encode:
            num_tweets = len(data)
            longest = 0
            tweets_tokenized = []
            labels_tokenized = []
            encode_data = tqdm(data)
            for tweet,keyphrase in encode_data:
                tweet = encoder.encode(tweet)
                keyphrase = encoder.encode(keyphrase)
                label = np.isin(tweet, keyphrase)
                longest = max(len(tweet), longest)
                tweets_tokenized.append(tweet)
                labels_tokenized.append(label)
                encode_data.set_postfix(encoding=data_type)

            data_placeholder = np.zeros((num_tweets, longest))
            label_placeholder = np.zeros((num_tweets, longest))

            for i in range(num_tweets):
                tweet = tweets_tokenized[i]
                label = labels_tokenized[i]
                data_placeholder[i, :len(tweet)] = tweet
                label_placeholder[i, :len(label)] = label

            if data_type == 'train' or data_type == 'test':
                self.data = np.stack([data_placeholder, label_placeholder])
            if data_type == 'val':
                self.val = np.stack([data_placeholder, label_placeholder])

        self.cache_path.mkdir(parents=True, exist_ok=True) 
        if self.is_train:
            data = {'train': self.data, 'val': self.val}
            fobject = open( str(self.cache_train), "wb" )
            pickle.dump(data, fobject)
            fobject.close()
        else:
            data = {'test': self.data}
            fobject = open( str(self.cache_test), "wb" )
            pickle.dump(data, fobject)
            fobject.close()
        self.pre_encoded = True

    def prepare_batch(self, data, batch_size, batch_i, use_tokenized, size):

        batch_begin = batch_size*batch_i
        batch_end   = batch_size*(batch_i+1) if batch_size*(batch_i+1) < size else size

        if use_tokenized:
            batch = data[:, batch_begin:batch_end, :]
            cutoff_idx = np.argmin(np.sum(batch[0], axis=0))
            batch = batch[:,:, :cutoff_idx]
        else:
            batch = [x[0] for x in data[batch_begin:batch_end*(i+1)]], [x[1] for x in data[batch_begin:batch_end*(i+1)]]
        return batch[0], batch[1]


    def size(self):
        if self.pre_encoded:
            _, size, _ = self.data.shape
            if self.is_train:
                _, val_size, _ = self.val.shape
                return size, val_size
        else:
            size = len(self.data)
            if self.is_train:
                val_size = len(self.val)
                return size, val_size
        return (size, )


    def data_iterator(self, data_type='train', batch_size=128):
        """Returns a generator that yields batches data with tags.
        Args:
            data_type: (str) flag in ['train', 'test', 'val'] giving which data to iterate over
            batch_size: (int)

        Yields:
            batch_data: (np.array) shape: (batch_size, max_len)
            batch_tags: (np.array) shape: (batch_size, max_len)
        """
        data = self.data
        size = self.size()[0]

        if self.is_train and data_type=='val':
            data = self.val  
            size = self.size()[1]

        for i in range(0,size//batch_size):
            yield self.prepare_batch(data=data, batch_size=batch_size, batch_i=i, use_tokenized=self.pre_encoded, size=size)


