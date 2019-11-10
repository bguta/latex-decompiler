import numpy as np
import pandas as pd
import sympy as sy
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms.functional as tv
from model.model import im2latex
from model.trainer import Trainer
from data.generator import generator
from sympy import preview
import matplotlib.pyplot as plt
from torchsummary import summary


def decode(token_ids, vocab_list):
    tokens = [vocab_list[x] for x in token_ids]
    decoded_string = ' '.join(tokens)
    return decoded_string.strip()

def decode_equation(encoding, vocab_list):
    enc_arr = encoding.squeeze().detach().numpy()
    if len(enc_arr.shape) > 1:
        token_ids = np.argmax(np.log(enc_arr), axis=-1)
        return decode(token_ids, vocab_list)
    return decode(enc_arr, vocab_list)

def beam_search_decoder(data, k):
    sequences = [[ [], 1.0 ]]
    # walk over each step in sequence
    for row in data:
        all_candidates = []
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -np.log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda x:x[1])
        # select k best
        sequences = ordered[:k]
    return sequences

def plot_att(img, atts, predicted, reshape=(6, 102)):
    for index, att in enumerate(atts):
        #print(f'{predicted[index]}')
        plt.axis('off')
        im_plt = plt.imshow(np.squeeze(img))
        plt.axis('off')
        plt.imshow(np.resize(np.squeeze(att.detach().numpy()), reshape), cmap='gray', alpha=0.6, extent=im_plt.get_extent())
        plt.title(f'Predicted {predicted[index]}')
        plt.show()

def preprocess_tex(tex):
    split = tex.split('<end>')
    processed_tex = split[0].strip()
    return processed_tex

def show_tex(tex):
    preview(f'${tex}$', viewer='file', filename='output.png', euler=False)

def encode_equation(string, vocab_list, dim, is_for_loss=False):
    encoding = [vocab_list.index(x) if x in vocab_list else print("ERROR") for x in string.split(' ')]
    if not is_for_loss:
        encoding.insert(0, len(vocab_list) - 2) # insert the start token
    encoding += [len(vocab_list) - 1] # insert the end token
    encoding += [0]*(dim - len(encoding)) # pad the rest
    return np.array(encoding)

# hyperparameters + files
DATA_DIR = 'data/'
IMAGE_DIR = DATA_DIR + 'images/'
DATASET = 'dataset.csv'
MODEL_DIR = DATA_DIR + 'saved_model/'
VOCAB = 'vocab.txt'
load_saved_model = False
max_equation_length = 200 + 2


# import the equations + image names and the tokens
dataset = pd.read_csv(DATA_DIR+DATASET)
vocabFile = open(DATA_DIR+VOCAB, 'r', encoding="utf8")
vocab_tokens = [x.replace('\n', '') for x in vocabFile.readlines()]
vocabFile.close()
vocab_size = len(vocab_tokens)
dataset['Y'] =  dataset['latex_equations'].apply(lambda x: encode_equation(x, vocab_tokens, max_equation_length, False))
dataset['Y_loss'] = dataset['latex_equations'].apply(lambda x: encode_equation(x, vocab_tokens, max_equation_length, True))
train_idx, val_idx = train_test_split(
    dataset.index, random_state=92372, test_size=0.20
)

# the validation and training data generators
train_generator = generator(list_IDs=train_idx,
            df=dataset,
            base_path=IMAGE_DIR,
            shuffle=True)

val_generator = generator(list_IDs=val_idx,
            df=dataset,
            base_path=IMAGE_DIR,
            shuffle=True)

# the model
model = im2latex(vocab_size)
if load_saved_model:
    print('Loading weights')
    checkpoint = torch.load(MODEL_DIR + 'best_ckpt.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
model.eval()