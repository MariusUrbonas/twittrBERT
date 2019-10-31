# <p align=center>`TwittrBERT`</p>
`TwittrBERT` is a `BERT` model trained on twitter data text. 
This repository currenlty provides a framework for training and tuning your own `TwittrBERT` for keyfrase extraction tasks, a traned model will be added in the future.

* It results in matching state-of-the-art performance on keyphrase extraction from twitter data. The details of the evaluation will be added, but more experiments are required. Evaluation code is included in this repo. 

There is no need to train from scratch, the original `BERT` already contains a lot of useful knowledge about the structure of the language. However one must account for the domain-specific features such as short informal sentences, emojis, misspellings and erratic punctuation. 

The approach taken my work was to retune the language model on a large corpus in an unsupervised fashion, and then train a token classifier head with a small set of supervised examples. In my experiments, this proved to improve the results.


### Training your `TwittrBert` model 

This project uses PyTorch, you will need [Hugging Face's repo](https://github.com/huggingface/pytorch-pretrained-BERT) where detailed instructions on using BERT models are provided. 

To run experiments you need to first setup the Python 3.6 environment:

* Pregenerate finetuning data (for the unsupervised learning):
```
python pregenerate_training_data.py --train_corpus [PAHT_TO_FILE: str] --output_dir lm_training/ --num_workers [NUM_WORKERS: int] --max_seq_len [YOUR_SEQ_LEN: int] --max_predictions_per_seq [MAX_PRED: int] --bert_model [BERT_MODEL]
```
* To finetune the LM use:
```
python finetune_on_pregenerated.py --pregenerated_data ../lm_training/ --output_dir ./temp/ --bert_model [BERT_MODEL]
```

* Then finally to training:
```
python train.py
```

For relevant flags please check the code.





