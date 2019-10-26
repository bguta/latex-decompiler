import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import sys
# we would like to be in the src directory to have access to main files
sys.path.append("..")

import numpy as np
import tensorflow as tf


'''
A reimplementation of model_v1 using tensorflow class models

'''
class BahdanauAttention(tf.keras.Model):
    '''
    src = https://www.tensorflow.org/tutorials/text/image_captioning
    '''
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 4, embedding_dim)
        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # score shape == (batch_size, 4, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, 4, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    '''
    Encoder to take in an image and return the feature vector
    '''
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        #self.conv_1         = tf.keras.layers.Conv2D(3, kernel_size=1, padding='same')
        self.img_model      = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=tf.keras.layers.Input(shape=(128, 1024, 3)))
        self.reshape        = tf.keras.layers.Reshape((4, 512*32), name='reshape_features')
        self.row_encoder    = tf.keras.layers.CuDNNGRU(embedding_dim, return_sequences=True,
                                       return_state=False,
                                       recurrent_initializer='glorot_uniform')
        #self.fc             = tf.keras.layers.Dense(embedding_dim, activation='relu')
        for layer in self.img_model.layers:
            layer.trainable = False
        

    def call(self, img):
        #img = self.conv_1(img)
        img = self.img_model(img)
        img = self.reshape(img)
        img = self.row_encoder(img)
        return img

class RNN_Decoder(tf.keras.Model):
    '''
    src = https://www.tensorflow.org/tutorials/text/image_captioning
    '''
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.CuDNNGRU(self.units, return_sequences=False,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        #self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc1 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)
    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)
        # x shape == (batch_size * max_length, hidden_size)
        #x = tf.reshape(x, (-1, x.shape[2]))
        # output shape == (batch_size * max_length, vocab)
        #x = self.fc2(x)
        return x, state, attention_weights
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))