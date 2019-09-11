import os
import torch
import pickle
import numpy as np

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


def load_checkpoint(checkpoint_folder, checkpoint_name):
    filepath = os.path.join(checkpoint_folder, checkpoint_name)

    print("<<< Loading checkpoint from ", filepath)

    checkpoint = torch.load(filepath)
    return checkpoint


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
        if not os.path.exists(self.save_path):
            os.mkdir(save_dir)
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
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

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
        #mask = batch_mask[0].cpu().numpy()
        mask = batch_mask[0]
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

