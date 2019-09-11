import numpy as np
import torch
import random


class DataLoader():

    def __init__(self, path_to_train, path_to_test, val_ratio=0.1, seed=42, shuffle=False):
        self.seed = seed
        self.train, self.val = self.load(path=path_to_train, split_val=True, val_ratio=val_ratio, shuffle=shuffle)
        self.test = self.load(path=path_to_test, split_val=False)
        #self.train_size, self.val_size, self.test_size = len(self.train), len(self.val), len(self.test)
        self.train_tokenized, self.val_tokenized, self.test_tokenized = None, None, None


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


    def pretokenize(self, tokenizer):
        print("Pretokenizing datasets")
        for data_type, data in [('train', self.train[:500]),('val', self.val[:500]), ('test', self.test)]:
            print(">>> Tokenizing {} dataset".format(data_type))
            num_tweets = len(data)
            longest = 0
            tweets_tokenized = []
            labels_tokenized = []
            for tweet,keyphrase in data:
                tweet = tokenizer.encode(tweet)
                keyphrase = tokenizer.encode(keyphrase)
                label = np.isin(tweet, keyphrase)
                longest = max(len(tweet), longest)
                tweets_tokenized.append(tweet)
                labels_tokenized.append(label)

            data_placeholder = np.zeros((num_tweets, longest))
            label_placeholder = np.zeros((num_tweets, longest))

            for i in range(num_tweets):
                tweet = tweets_tokenized[i]
                label = labels_tokenized[i]
                data_placeholder[i, :len(tweet)] = tweet
                label_placeholder[i, :len(label)] = label

            if data_type == 'train':
                self.train_tokenized = np.stack([data_placeholder, label_placeholder])
            if data_type == 'val':
                self.val_tokenized = np.stack([data_placeholder, label_placeholder])
            if data_type == 'test':
                self.test_tokenized = np.stack([data_placeholder, label_placeholder])


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


    def data_iterator(self, data_type='train', batch_size=128):
        """Returns a generator that yields batches data with tags.
        Args:
            data_type: (str) flag in ['train', 'test', 'val'] giving which data to iterate over
            batch_size: (int)

        Yields:
            batch_data: (np.array) shape: (batch_size, max_len)
            batch_tags: (np.array) shape: (batch_size, max_len)
        """
        if self.train_tokenized is None:
            print(">>> Data iterator was not provided tokenizer, iterating over untokenized data!")
            use_tokenized = False
            if data_type == 'train':
                data = self.train
            elif data_type == 'val':
                data = self.val
            elif data_type == 'test':
                data = self.test
            else:
                raise ValueError(_type)
        else:
            use_tokenized = True
            if data_type == 'train':
                data = self.train_tokenized
            elif data_type == 'val':
                data = self.val_tokenized
            elif data_type == 'test':
                data = self.test_tokenized
            else:
                raise ValueError(_type)

        if use_tokenized:
            _, size, _ = data.shape
        else:
            size = len(data)

        for i in range(0,size//batch_size):
            yield self.prepare_batch(data=data, batch_size=batch_size, batch_i=i, use_tokenized=use_tokenized, size=size)


