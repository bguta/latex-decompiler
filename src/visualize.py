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
from model.model_v1 import im2latex
from data.dataGenerator import generator
from scipy.special import softmax
import tensorflow.keras.backend as K
from model.model_v2 import BahdanauAttention, CNN_Encoder, RNN_Decoder
#tf.enable_eager_execution()

def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup:tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences


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

def make_prediction(img, seq, vocab, max_len=661):
    copy = np.zeros((1,max_len))
    copy[0][0] = seq[0][0]
    atts = []
    yp = np.expand_dims(seq[0][0],axis=0)
    model.reset_state()
    for i in range(1,max_len):
        yp, att = model.predict([img, yp])
        yp = np.argmax(softmax(yp, axis=-1),axis=-1)
        atts.append(att)
        copy[0][i] = yp
        if yp == len(vocab) - 1:
            break
    return copy, atts

def make_prediction2(inputs, target, vocab, max_len=661):
    hidden = decoder.reset_state(batch_size=1)
    atts = []
    img, seq = inputs
    yp = tf.expand_dims(target[0],axis=0)
    features = encoder(img)
    equation = np.zeros((1,max_len))
    equation[0][0] = yp[0][0]
    for i in range(1,max_len):
        yp, hidden, att_weights = decoder(yp, features, hidden)
        yp = np.argmax(softmax(yp.numpy(), axis=-1), axis=-1)
        equation[0][i] = yp
        atts.append(tf.reshape(att_weights,  (-1, )).numpy())
        if yp == len(vocab) - 1:
            break
    return equation, atts

def plot_att(img, atts, reshape=(4, 32)):
    for index, att in enumerate(atts):
        print(f'{index}')
        im_plt = plt.imshow(np.squeeze(img), cmap='gray')
        plt.imshow(np.resize(att, reshape), cmap='gray', alpha=0.6, extent=im_plt.get_extent())
        plt.show()

def show_latex(tex):
    preview(f'${tex}$', viewer='file', filename='output.png', euler=False)

def preprocess(x):
    return x/255.

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
encoder_lstm_units = 128
decoder_lstm_units = 256

# import the equations + image names and the tokens
dataset = pd.read_csv(DATA_DIR+DATASET)
#dataset = dataset.head(10000)
vocabFile = open(DATA_DIR+VOCAB, 'r', encoding="utf8")
vocab_tokens = [x.replace('\n', '') for x in vocabFile.readlines()]
vocabFile.close()
vocab_size = len(vocab_tokens)
train_idx, val_idx = train_test_split(
    dataset.index, random_state=876, test_size=0.20
)

# encoder = CNN_Encoder(256)
# decoder = RNN_Decoder(256, 659+2, vocab_size)
# optimizer = tf.keras.optimizers.Adam()

# checkpoint_path = MODEL_DIR
# ckpt = tf.train.Checkpoint(encoder=encoder,
#                            decoder=decoder,
#                            optimizer=optimizer)
# status = ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))

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
latex_model = im2latex(encoder_lstm_units, decoder_lstm_units, vocab_tokens)
model = latex_model.model
if load_saved_model:
    print('Loading weights')
    model.load_weights(MODEL_DIR + latex_model.name + '.h5', by_name=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])