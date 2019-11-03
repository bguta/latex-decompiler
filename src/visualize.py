import numpy as np
import pandas as pd
import sympy as sy
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F
from model.model import im2latex
from model.trainer import Trainer
from data.dataGenerator import generator
from sympy import preview
import matplotlib.pyplot as plt


def decode(token_ids, vocab_list):
    tokens = [vocab_list[x] for x in token_ids]
    decoded_string = ' '.join(tokens)
    return decoded_string.strip()

def decode_equation(encoding, vocab_list):
    token_ids = np.argmax(encoding, axis=-1)
    return decode(token_ids, vocab_list)

def preprocess_tex(tex):
    split = tex.split('<end>')
    processed_tex = split[0].strip()
    return processed_tex

def show_latex(tex):
    preview(f'${tex}$', viewer='file', filename='output.png', euler=False)


def preprocess(x):
    return x/255.

# hyperparameters + files
DATA_DIR = 'data/'
IMAGE_DIR = DATA_DIR + 'images/'
DATASET = 'dataset.csv'
MODEL_DIR = DATA_DIR + 'saved_model/'
VOCAB = 'vocab_8k.txt'
BATCH_SIZE = 1
EPOCHS = 1
START_EPOCH = 0
IMAGE_DIM = (32, 416)
load_saved_model = True
max_equation_length = 659


# import the equations + image names and the tokens
dataset = pd.read_csv(DATA_DIR+DATASET)
vocabFile = open(DATA_DIR+VOCAB, 'r', encoding="utf8")
vocab_tokens = [x.replace('\n', '') for x in vocabFile.readlines()]
vocabFile.close()
vocab_size = len(vocab_tokens)
train_idx, val_idx = train_test_split(
    dataset.index, random_state=2312, test_size=0.20
)

# the validation and training data generators
train_generator = generator(list_IDs=train_idx,
            df=dataset,
            dim=IMAGE_DIM,
            eq_dim=max_equation_length,
            batch_size=BATCH_SIZE,
            base_path=IMAGE_DIR,
            preprocess=preprocess,
            vocab_list=vocab_tokens,
            shuffle=True,
            n_channels=3)

val_generator = generator(list_IDs=val_idx,
            df=dataset,
            dim=IMAGE_DIM,
            eq_dim=max_equation_length,
            batch_size=BATCH_SIZE,
            base_path=IMAGE_DIR,
            preprocess=preprocess,
            vocab_list=vocab_tokens,
            shuffle=True,
            n_channels=3)

# the model
model = im2latex(vocab_size)
if load_saved_model:
    print('Loading weights')
    checkpoint = torch.load(MODEL_DIR + 'best_ckpt.pt')
    model.load_state_dict(checkpoint['model_state_dict'])