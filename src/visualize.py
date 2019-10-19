import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings

import tensorflow as tf
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from sympy import preview
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.model import im2latex
from data.dataGenerator import generator
from scipy.special import softmax


def encode_equation(string, vocab_list, dim):
    encoding = np.zeros((*dim))

    for i, token in enumerate(string.split(' ')):
        if token in vocab_list:
            encoding[i][vocab_list.index(token)] = 1.0
    return encoding

def decode_equation(encoding, vocab_list):
    token_ids = np.argmax(encoding, axis=-1)
    tokens = [vocab_list[x] for x in token_ids]
    decoded_string = ' '.join(tokens)
    return decoded_string.strip()

def decode(enc, vocab):
    ids = enc.astype(int)
    tokens = [vocab[x] for x in ids]
    decoded_string = ' '.join(tokens)
    return decoded_string.strip()

def make_prediction(img, seq, vocab):
    max_len = len(seq[0])
    copy = np.array(seq)
    for i in range(1,max_len):
        yp = np.argmax(softmax(model.predict([img, copy]), axis=-1),axis=-1)
        copy[0][i] = yp
        if yp == len(vocab) - 1:
            break
    return copy

def show_latex(tex):
    preview(f'${tex}$', viewer='file', filename='output.png', euler=False)

def preprocess(x):
    return x

# hyperparameters + files
DATA_DIR = 'data/'
IMAGE_DIR = DATA_DIR + 'images/'
DATASET = 'dataset.csv'
MODEL_DIR = DATA_DIR + 'saved_model/'
VOCAB = 'vocab_50k.txt'
BATCH_SIZE = 1
EPOCHS = 1
START_EPOCH = 0
IMAGE_DIM = (128, 1024)
load_saved_model = True
max_equation_length = 659 + 2
encoder_lstm_units = 256
decoder_lstm_units = 512

# import the equations + image names and the tokens
dataset = pd.read_csv(DATA_DIR+DATASET)
#dataset = dataset.head(10)
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
            target_df=dataset,
            dim=IMAGE_DIM,
            eq_dim=(max_equation_length, vocab_size),
            batch_size=BATCH_SIZE,
            base_path=IMAGE_DIR,
            preprocess=preprocess,
            vocab_list=vocab_tokens,
            shuffle=True,
            n_channels=1)

val_generator = generator(list_IDs=val_idx,
            df=dataset,
            target_df=dataset,
            dim=IMAGE_DIM,
            eq_dim=(max_equation_length, vocab_size),
            batch_size=BATCH_SIZE,
            base_path=IMAGE_DIR,
            preprocess=preprocess,
            vocab_list=vocab_tokens,
            shuffle=True,
            n_channels=1)

# the model
latex_model = im2latex(encoder_lstm_units, decoder_lstm_units, vocab_tokens, max_equation_length)
model = latex_model.model
if load_saved_model:
    print('Loading weights')
    model.load_weights(MODEL_DIR + latex_model.name + '.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])