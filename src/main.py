import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import tensorflow as tf
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import sympy as sy
from sklearn.model_selection import train_test_split
from create_data import create_data
from model.model import im2latex
from model.metrics import ce, acc, acc_full, focal_loss
from data.dataGenerator import generator
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
#import segmentation_models as sm

def create_dataset():
    creator = create_data(image_size=(128,1024), 
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


def encode_equation(string, vocab_list, dim):
    encoding = np.zeros((dim[0], dim[1]))
    tokens = string.split(' ')
    tokens_length = len(tokens)

    for i in range(dim[0]):
        if i < tokens_length:
            if tokens[i] in vocab_list:
                encoding[i][vocab_list.index(tokens[i])] = 1.0
            else:
                encoding[i][0] = 1.0
        else:
            encoding[i][0] = 1.0
    return encoding

def decode_equation(encoding, vocab_list):
    token_ids = np.argmax(encoding, axis=-1)
    tokens = [vocab_list[x] for x in token_ids]
    decoded_string = ' '.join(tokens)
    return decoded_string

def preprocess(x):
    return x/255.

def train():
    # hyperparameters + files
    DATA_DIR = 'data/'
    IMAGE_DIR = DATA_DIR + 'images/'
    DATASET = 'dataset.csv'
    MODEL_DIR = DATA_DIR + 'saved_model/'
    VOCAB = 'vocab_50k.txt'
    BATCH_SIZE = 2
    EPOCHS = 10
    START_EPOCH = 0
    IMAGE_DIM = (128, 1024)
    load_saved_model = False
    max_equation_length = 659
    encoder_lstm_units = 256
    decoder_lstm_units = 256
    gamma = 2.
    alpha = 4.


    # import the equations + image names and the tokens
    dataset = pd.read_csv(DATA_DIR+DATASET)
    #smaller = dataset.head(20)
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
    
    # initialize our model
    latex_model = im2latex(encoder_lstm_units, decoder_lstm_units, vocab_tokens, max_equation_length)
    model = latex_model.model
    if load_saved_model:
        print('Loading Weights')
        model.load_weights(MODEL_DIR + latex_model.name + '.h5')
    model.compile(optimizer=Adam(lr=1e-3), loss=focal_loss(gamma=gamma, alpha=alpha), metrics=[acc, acc_full])
    model.summary()
    input()

    # callbacks
    nan_stop = TerminateOnNaN()
    checkpoint = ModelCheckpoint(
        MODEL_DIR + latex_model.name + '.h5',
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='max'
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.7,
                              patience=2, min_lr=1e-8, verbose=1)

    # train
    history = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        callbacks=[nan_stop, checkpoint, reduce_lr],
        workers=3,
        epochs=EPOCHS,
        use_multiprocessing=True,
        initial_epoch=0
    )

if __name__ == '__main__':
    #create_dataset()
    train()
