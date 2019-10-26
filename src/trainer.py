import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import sys
import time
# we would like to be in the src directory to have access to main files
sys.path.append("..")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.dataGenerator import generator
import tensorflow as tf
import tensorflow.keras.backend as K
import os
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model.model_v2 import BahdanauAttention, CNN_Encoder, RNN_Decoder
tf.enable_eager_execution()


def preprocess(x):
    return x

DATA_DIR = 'data/'
IMAGE_DIR = DATA_DIR + 'images/'
DATASET = 'dataset.csv'
MODEL_DIR = DATA_DIR + 'saved_model/'
VOCAB = 'vocab_50k.txt'
BATCH_SIZE = 1
IMAGE_DIM = (128, 1024)
max_equation_length = 659 + 2
embedding_dim = 256
units         = 512
batch_size = 8
#vocab_size    = 659 + 2
EPOCHS = 5
start_epoch = 0


# import the equations + image names and the tokens
dataset = pd.read_csv(DATA_DIR+DATASET)
dataset = dataset.head(1000)
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

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)


optimizer = tf.keras.optimizers.Adam(lr=1e-1)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

checkpoint_path = MODEL_DIR
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

@tf.function
def loss_function(real, pred):
    #mask = tf.math.logical_not(tf.math.equal(real, 0))
    gt = tf.cast(real, tf.int32)
    #loss_ = loss_object(gt, pred) #loss_object(real, pred)
    # mask = tf.cast(mask, dtype=loss_.dtype)
    # loss_ *= mask
    return tf.reduce_mean(loss_object(gt, pred))
@tf.function
def acc_function(real, pred):
    pr = tf.argmax(pred, axis=-1)
    gt = tf.cast(real, tf.int64)
    return K.mean(K.equal(gt, pr))

@tf.function
def val_step(img, hidden, target, dec_input):
    acc = 0.0
    #i = 1
    features = encoder(img)
    #dec_input = tf.expand_dims(target[0],axis=0)
    for i in tf.range(1, target.shape[0]):
        # passing the features through the decoder
        predictions, hidden, _ = decoder(dec_input, features, hidden)
        acc += acc_function(target[i], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(tf.argmax(tf.nn.softmax(predictions, axis=-1), axis=-1), axis=-1)
        dec_input = tf.cast(dec_input, tf.float32)
    total_acc = (acc / int(target.shape[0]))
    return acc, total_acc, hidden, dec_input

@tf.function
def train_step(img, hidden, target):
    loss = 0.0
    #i = 1
    with tf.GradientTape() as tape:
        features = encoder(img)
        dec_input = tf.expand_dims(tf.expand_dims(target[0],axis=0), axis=1)
        for i in tf.range(1, target.shape[0]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[i], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(tf.expand_dims(target[i],axis=0), axis=1)

        total_loss = (loss / int(target.shape[0]))
        trainable_variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss, total_loss, hidden

loss_plot = []
val_loss_plot = []
tf.print('Starting to Train')
with tf.device('GPU:0'):
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss, total_val_acc, total_acc = 0, 0, 0
        steps, val_steps = 0, 0
        for (batch, (inputs, target)) in enumerate(train_generator):
            steps += target.shape[0]
            batch_acc, batch_loss = 0, 0
            hidden = decoder.reset_state(batch_size=1)
            img = inputs
            num_splits = target.shape[0]//batch_size
            if num_splits > 0:
                states = np.array_split(target, num_splits)
            else:
                states = [target]

            for state in states:
                #K.clear_session()
                b_loss, t_loss, hidden = train_step(img, hidden, state)
                total_loss += t_loss
                batch_loss += b_loss
            
            hidden = decoder.reset_state(batch_size=1)
            dec_input = tf.expand_dims(tf.expand_dims(target[0],axis=0), axis=1)
            for state in states:
                #K.clear_session()
                b_acc, t_acc, hidden, dec_input = val_step(img, hidden, state, dec_input)
                total_acc += t_acc
                batch_acc += b_acc

            if batch % 30 == 0:
                print(f'Epoch: {epoch + 1} | Batch: {batch} | Loss: {batch_loss / int(target.shape[0])} | Acc: {batch_acc / int(target.shape[0])}', end='\r')
                
        # storing the epoch end loss value to plot later
        #loss_plot.append(total_loss / steps)
        if epoch % 1 == 0:
            ckpt_manager.save()

        for (batch, (inputs, target)) in enumerate(val_generator):
            val_steps += target.shape[0]
            hidden = decoder.reset_state(batch_size=1)
            dec_input = tf.expand_dims(tf.expand_dims(target[0],axis=0), axis=1)
            img = inputs
            num_splits = target.shape[0]//batch_size
            if num_splits > 0:
                states = np.array_split(target, num_splits)
            else:
                states = [target]
            for state in states:
                #K.clear_session()
                _, t_acc, hidden, dec_input = val_step(img, hidden, state, dec_input)
                total_val_acc += t_acc
        
        print('Epoch: {} | Loss: {:.6f} | Acc: {:.6f} | Val_Acc: {:.6f}'.format(epoch + 1,
                                             total_loss/steps, total_acc / steps, total_val_acc/val_steps ))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        train_generator.on_epoch_end()
        val_generator.on_epoch_end()