import numpy as np
import pandas as pd
import sympy as sy
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from model.model import im2latex
from model.trainer import Trainer
from model.metrics import loss_fn
from data.generator import generator
from torch.utils.data import DataLoader

def create_dataset():
    creator = create_data(image_size=(32, 416), 
                output_csv='data/dataset.csv', 
                output_dir='data/images', 
                formula_file='data/formulas.txt')
    creator.create()

def create_vocabset():
    dataset = pd.read_csv('data/dataset.csv')
    eqs = [string.split(' ') for string in dataset.latex_equations.values.tolist()]
    all_tokens = []
    for token_set in eqs:
        all_tokens += token_set
    
    unique_tokens = np.insert(np.unique(all_tokens), 0, '')
    with open('vocab_.txt', 'w') as f:
        f.write('\n'.join(unique_tokens))

def preprocess(x):
    return x/255.

def train():
    # hyperparameters + files
    DATA_DIR = 'data/'
    IMAGE_DIR = DATA_DIR + 'images/'
    DATASET = 'dataset.csv'
    MODEL_DIR = DATA_DIR + 'saved_model'
    VOCAB = 'vocab_8k.txt'
    BATCH_SIZE = 8
    EPOCHS = 10
    START_EPOCH = 0
    IMAGE_DIM = (32, 416)
    load_saved_model = True
    max_equation_length = 232 + 2

    # import the equations + image names and the tokens
    dataset = pd.read_csv(DATA_DIR+DATASET)
    #dataset = dataset.head(100)
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
    
    # initialize our model
    model = im2latex(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    lr_schedule = None
    epsilon = 1.0
    best_val_loss = np.inf
    if load_saved_model:
        checkpoint = torch.load(MODEL_DIR + '/ckpt-32-0.6236.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epsilon = checkpoint['epsilon']
        print('Loaded weights')

    trainer = Trainer(optimizer=optimizer,
                    loss_fn=loss_fn,
                    model=model,
                    train_generator=train_generator,
                    val_generator=val_generator,
                    model_path=MODEL_DIR,
                    lr_scheduler=lr_schedule,
                    init_epoch=START_EPOCH,
                    epsilon=epsilon,
                    best_val_loss=best_val_loss,
                    num_epochs=EPOCHS)
    trainer.train()

if __name__ == '__main__':
    train()
