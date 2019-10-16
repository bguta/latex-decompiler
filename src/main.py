import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import tensorflow as tf
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import sympy as sy
from sklearn.model_selection import train_test_split
from create_data import create_data
from model.model import im2latex
from model.metrics import ce, acc
from data.dataGenerator import generator
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import segmentation_models as sm

def create_dataset():
    creator = create_data(image_size=(128,1024), 
                output_csv='data/dataset.csv', 
                output_dir='data/images', 
                formula_file='data/formulas.txt')
    creator.create()

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
    VOCAB = 'vocab_995.txt'
    BATCH_SIZE = 16
    EPOCHS = 5
    START_EPOCH = 0
    IMAGE_DIM = (128, 1024)
    load_saved_model = True
    max_equation_length = 314
    encoder_lstm_units = 256
    decoder_lstm_units = 256


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
        model.load_weights(MODEL_DIR + latex_model.name + '.h5')
    model.compile(optimizer=Adam(lr=1e-2), loss=sm.losses.CategoricalFocalLoss(alpha=1.0, gamma=2.0), metrics=[acc])
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

    # train
    history = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        callbacks=[nan_stop, checkpoint],
        workers=3,
        epochs=EPOCHS,
        use_multiprocessing=True,
        initial_epoch=0
    )

if __name__ == '__main__':
    train()
