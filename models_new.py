import torch
import torch.nn as nn
from pytorch_transformers import *
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from tqdm import tqdm
import numpy as np


class BertForKeyprhaseExtraction(nn.Module):

	def __init__(self, arch, checkpoint=None):
		pass

	@classmethod
	def from_pretrained(cls, path):
		print("<<< Loading checkpoint from ", filepath)
	    checkpoint = torch.load(filepath)
	    return cls()