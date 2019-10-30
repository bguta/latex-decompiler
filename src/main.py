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
from model.model_v1 import im2latex
from model.metrics import ce, acc, acc_full
from data.dataGenerator import generator
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam, Nadam
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
tf_session = K.get_session()
tf_graph = tf.get_default_graph()
#import segmentation_models as sm

def create_dataset():
    creator = create_data(image_size=(32,416), 
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
    with open('vocab_8k.txt', 'w') as f:
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

def generate_data(generator, model):
    epoch_len = len(generator)
    batch_size = 1
    i = 0
    while True:
        X,y = generator.__getitem__(i)
        steps = len(y)
        batchs = steps // batch_size
        if steps == 0:
            i = 0
            generator.on_epoch_end()
            continue
        if batchs <= 0:
            batchs = 1
        imgs = np.array_split(X[0], batchs)
        #states = np.array_split(X[1], batchs)
        last_out = np.array_split(X[1], batchs)
        outputs = np.array_split(y, batchs)

        for j in range(len(imgs)):
            yield ([ imgs[j], last_out[j] ], outputs[j])
        if j == len(imgs) - 1:
            i += 1
            with tf_session.as_default():
                with tf_graph.as_default():
                    model.reset_states()
def train():
    # hyperparameters + files
    DATA_DIR = 'data/'
    IMAGE_DIR = DATA_DIR + 'images/'
    DATASET = 'dataset.csv'
    MODEL_DIR = DATA_DIR + 'saved_model/'
    VOCAB = 'vocab_8k.txt'
    BATCH_SIZE = 1
    EPOCHS = 130
    START_EPOCH = 0
    IMAGE_DIM = (32,416)
    load_saved_model = False
    max_equation_length = 659 + 2 # 659 tokens + the 2 start/end tokens
    encoder_lstm_units = 256
    decoder_lstm_units = 512


    # import the equations + image names and the tokens
    dataset = pd.read_csv(DATA_DIR+DATASET)
    dataset = dataset.head(20)
    vocabFile = open(DATA_DIR+VOCAB, 'r', encoding="utf8")
    vocab_tokens = [x.replace('\n', '') for x in vocabFile.readlines()]
    vocabFile.close()
    vocab_size = len(vocab_tokens)
    train_idx, val_idx = train_test_split(
        dataset.index, random_state=876, test_size=0.20
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
    latex_model = im2latex(encoder_lstm_units, decoder_lstm_units, vocab_tokens)
    model = latex_model.model
    if load_saved_model:
        print('Loading Model')
        model.load_weights(MODEL_DIR + latex_model.name + '.h5', by_name=True)
    model.compile(optimizer=Nadam(lr=1e-4), loss=ce, metrics=[acc])
    model.summary()
    #plot_model(model, to_file='model.png')
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
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.95,
                              patience=3, min_lr=1e-10, verbose=1)
    def scheduler(epoch, lr):
        return lr*0.95       
    lr_schedule = LearningRateScheduler(scheduler, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    csv_logger = CSVLogger(MODEL_DIR+'training_log.csv', append=True)
    # train
    history = model.fit_generator(
        generate_data(train_generator, model),
        steps_per_epoch=1000,
        validation_data=generate_data(val_generator, model),
        validation_steps=500,
        callbacks=[nan_stop, checkpoint, reduce_lr, csv_logger, early_stopping],
        workers=1,
        epochs=EPOCHS,
        use_multiprocessing=False,
        initial_epoch=START_EPOCH
    )

if __name__ == '__main__':
    #create_vocabset()
    #create_dataset()
    train()
