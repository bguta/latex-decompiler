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
from data.create_data import create_data
from torch.utils.data import DataLoader

def create_dataset():
    creator = create_data(image_size=(128, 416), 
                output_csv='data/dataset.csv', 
                output_dir='data/images',
                formula_file='data/normalized_.txt')
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

def encode_equation(string, vocab_list, dim, is_for_loss=False):
    encoding = [vocab_list.index(x) if x in vocab_list else print('ERROR') for x in string.split(' ')]
    if not is_for_loss:
        encoding.insert(0, len(vocab_list) - 2) # insert the start token
    encoding += [len(vocab_list) - 1] # insert the end token
    encoding += [0]*(dim - len(encoding)) # pad the rest
    return np.array(encoding)

def train():
    # hyperparameters + files
    DATA_DIR = 'data/'
    IMAGE_DIR = DATA_DIR + 'images/'
    DATASET = 'dataset.csv'
    MODEL_DIR = DATA_DIR + 'saved_model'
    VOCAB = 'vocab.txt'
    BATCH_SIZE = 16
    EPOCHS = 10
    START_EPOCH = 0
    IMAGE_DIM = (128, 416)
    load_saved_model = False
    max_equation_length = 200 + 2


    vocabFile = open(DATA_DIR+VOCAB, 'r', encoding="utf8")
    vocab_tokens = [x.replace('\n', '') for x in vocabFile.readlines()]
    vocabFile.close()
    vocab_size = len(vocab_tokens)
    # import the equations + image names and the tokens
    dataset = pd.read_csv(DATA_DIR+DATASET)
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

    train_generator = DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_generator = DataLoader(val_generator, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    # initialize our model
    model = im2latex(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=1, verbose=True, cooldown=1, min_lr=1e-7)
    epsilon = 1.0
    best_val_loss = np.inf
    if load_saved_model:
        checkpoint = torch.load(MODEL_DIR + '/best_ckpt.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        START_EPOCH = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        #epsilon = checkpoint['epsilon']
        print('Loaded weights')
    print(f'epsilon val: {epsilon}')
    print(f'best_val_loss: {best_val_loss}')
    print(f'model summary: {model}')
    input()

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