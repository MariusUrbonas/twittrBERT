import os
import torch
import pickle
import numpy as np
import re

class TweetProcesser:
    @staticmethod
    def replace_url(text):
        url_pattern = '(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]\
                    *[\w@?^=%&/~+#-])?(/([A-Z]|[a-z]|[0-9])*)?'
        return re.sub(url_pattern, 'URL', text)

    @staticmethod
    def replace_nl(text):
        return " ".join(text.split("\n"))

    @staticmethod
    def split_to_sentences(text):
        sentence_pattern = re.compile("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")
        return sentence_pattern.split(text)

    @staticmethod
    def long_enough(sentence, min_length):
        return len(sentence.split()) >= min_length

    @classmethod
    def merge_short_sentences(cls, sentences, min_length):
        # if one of the sentences in the list is too short for current theshold
        # merge it with shortest surrounding sentence
        merged_sentences = []
        i = 0
        last_merged = False
        while i < len(sentences)-1:
            last_merged = False
            if not cls.long_enough(sentences[i], min_length):
                sent = sentences[i] + " " + sentences[i+1]
                merged_sentences.append(sent)
                i += 1
                last_merged = True
            else:
                merged_sentences.append(sentences[i])
            i += 1
        if not last_merged:
            if cls.long_enough(sentences[i], min_length):
                merged_sentences.append(sentences[-1])
            else:
                merged_sentences = merged_sentences[:-1] +\
                                   [merged_sentences[-1]+' '+ sentences[-1]]
        return merged_sentences

    @classmethod
    def preprocess_tweet(cls, tweet_json, min_length, min_num, split):
        text = Tweet_parser.get_full_text(tweet_json)
        if text is None:
            return None
        text = cls.replace_url(text)
        text = cls.replace_nl(text)
        if split:
            sentences = cls.split_to_sentences(text)
        if len(sentences) == min_num:
            return sentences
        if len(sentences) < min_num:
            return None
        sentences = cls.merge_short_sentences(sentences, min_length)
        if len(sentences) < min_num:
            return None
        return sentences



def save_checkpoint(state, epoch, score, checkpoint, is_best, tag='experiment'):
    
    print(">>> Saving Checkpoint")

    name = '{}_epoch_{}_score_{}.pth.tar'.format(tag, epoch, score)
    if is_best:
        name = 'BEST_{}_epoch_{}_score_{}.pth.tar'.format(tag, epoch, score)
    filepath = os.path.join(checkpoint, name)

    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)


class Stats:

    def __init__(self, save_dir, tag):
        self.data = {}
        self.save_dir=save_dir
        name = "stats_{}.pickle".format(tag)
        self.save_path = os.path.join(save_dir, name)

    def update(self, metrics,  epoch, train_loss):
        self.data[epoch] = {}
        self.data[epoch]['f1'] =  metrics.f1()
        self.data[epoch]['val_loss'] = metrics.loss
        self.data[epoch]['train_loss'] = train_loss
        self.data[epoch]['precision'] = metrics.precision()
        self.data[epoch]['recall'] = metrics.recall()

    def save(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        with open(self.save_path, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
   

class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

class Params():

    def __init__(self):
        pass

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class Metrics:

    def __init__(self):
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.loss = 0

    def update(self, batch_pred, batch_true, batch_mask):
        mask = batch_mask
        self.true_pos += np.sum(batch_pred[mask] & batch_true[mask])
        self.false_pos += np.sum(batch_pred[mask] & np.logical_not(batch_true[mask]))
        self.false_neg += np.sum(np.logical_not(batch_pred[mask]) & batch_true[mask])

    def f1(self):
        if self.precision() == 0 and self.recall() == 0:
            self._f1=0
        else:
            self._f1 = 2*(self.precision()*self.recall()/(self.precision()+self.recall()))
        return self._f1
    
    def precision(self):
     	return self.true_pos/(self.true_pos+self.false_pos)

    def recall(self):
        return self.true_pos/(self.true_pos+self.false_neg)

