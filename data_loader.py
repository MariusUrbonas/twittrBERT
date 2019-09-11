import numpy as np
import torch

class DataLoader():

    def __init__(self, path_to_train, path_to_test, val_split=0.1, seed=42, device=None, load_emb=False):
        self.val_split = val_split
        self.train, self.val = self.load(path_to_train, True)
        self.test = self.load(path_to_test, False)
        self.train_size = len(self.train)
        self.val_size = len(self.val)
        self.test_size = len(self.test)
        self.seed = seed
        self.device = device


    def load(self, path: str, train=True):
        print("Loading --{}--\n\t from {} ....\n".format("TRAIN" if train else "TEST", path))

        keyphrases = []
        tweets = []

        with open(path) as infile:
            for line in infile:
                t, key = line.strip().split('\t')
                tweets.append(t)
                keyphrases.append(key)
                #print(tweets[-1])
        print("{} Data Loaded....\n".format(len(tweets)))
        if train:
            return self.split_validation(list(zip(tweets, keyphrases)))
        return list(zip(tweets, keyphrases))


    def split_validation(self, train_list):
        size = len(train_list)
        split = int(size*(1-self.val_split))
        #return train_list[:split],train_list[split:]
        return train_list, train_list[split:]


    def prepare_data(self, data, order, batch_size, i, tokenizer):
        batch = [data[idx] for idx in order[i*batch_size:(i+1)*batch_size]]

        batch_max_item_len = max([len(tokenizer.encode(s[0])) for s in batch])
        batch_len = len(batch)

        batch_data = np.zeros((batch_len, batch_max_item_len))
        batch_labels = np.zeros((batch_len, batch_max_item_len))

        for t,(tweet,keyphrase) in enumerate(batch):
            item = tokenizer.encode(tweet)
            keyphrase = tokenizer.encode(keyphrase)
            label = np.isin(item, keyphrase)

            batch_data[t,:len(item)] = item
            batch_labels[t,:len(item)] = label

        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_data = batch_data.to(device=self.device)
        batch_labels = batch_labels.to(device=self.device)
        return batch_data, batch_labels

    def data_iterator_from_presaved(self, _type='train',batch_size=16 ,tokenizer=None, shuffle=False):
        if _type == 'train':
            data = self.train
            n_chunks = 61
            n_start = 1
        elif _type == 'val':
            data = self.train
            n_start = 61
            n_chunks = 67
        elif _type == 'test':
            data = self.test
        else:
            raise ValueError(_type)
        size = len(data)
        order = list(range(size))
        chunk_size = 1024
        for i in range(n_start, n_chunks):
            data_chunk = np.load('train_data_batch_{}.npy'.format(i*chunk_size))
            #print()
            for j in range(data_chunk.shape[0]//batch_size):
                yield torch.tensor(data_chunk[j:j+batch_size]).cuda(), self.prepare_data(data, order, chunk_size, i, tokenizer)[0][j:j+batch_size], self.prepare_data(data, order, chunk_size, i, tokenizer)[1][j:j+batch_size]

    def data_iterator(self, _type='train', batch_size=128 ,tokenizer=None, shuffle=False):
        """Returns a generator that yields batches data with tags.
        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """
        if _type == 'train':
            data = self.train
        elif _type == 'val':
            data = self.val
        elif _type == 'test':
            data = self.test
        else:
            raise ValueError(_type)

        size = len(data)
        order = list(range(size))

        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        for i in range(1,size//batch_size):
            yield self.prepare_data(data, order, batch_size, i, tokenizer)
