import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import sys
# we would like to be in the src directory to have access to main files
sys.path.append("..")


from tensorflow.keras.layers import CuDNNLSTM, Input, Dense, RepeatVector, TimeDistributed, Reshape, Bidirectional, Dropout, Flatten, Permute, Activation, Multiply, Lambda
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from model.cnn import cnn
import numpy as np
import tensorflow.keras.backend as K

class im2latex:
    r""" Create the image to latex converter model.

    # Arguments

        encoder_lstm_units:     The dimensionality of the output space for encoder LSTM layers
        decoder_lstm_units:     The dimensionality of the output space for decoder LSTM layers
        vocab_list:             The array of possible outputs of the language model
        embedding_size:         The max length of the equation


    # Example

    .. code:: python
    latex_model = im2latx(encoder_lstm_units, decoder_lstm_units, vocab_list)

    model = latex_model.model
    """

    def __init__(self, encoder_lstm_units, decoder_lstm_units, vocab_list, embedding_size=512):
        self.name = 'im2latex'
        self.encoder_lstm_units = encoder_lstm_units
        self.decoder_lstm_units = decoder_lstm_units
        self.embedding_size = embedding_size
        self.vocab_list = vocab_list
        self.vocab_size = len(self.vocab_list)

        # encoder
        image_input = Input(shape=(128,1024,1))
        x = cnn(image_input)
        x = Reshape((8, 64*128))(x)

        language_model = CuDNNLSTM(self.encoder_lstm_units, return_sequences=True, kernel_initializer='he_uniform')(x)
        #language_model = CuDNNLSTM(self.encoder_lstm_units, return_sequences=False, kernel_initializer='he_uniform')(language_model)
        #language_model = Lambda(lambda xin: K.expand_dims(xin, axis=-1))(language_model)
        #language_model = CuDNNLSTM(self.encoder_lstm_units, return_sequences=True,  return_state=True, kernel_initializer='he_uniform')(language_model)
        #language_model = CuDNNLSTM(self.encoder_lstm_units, return_sequences=True, kernel_initializer='he_uniform')(language_model)

        attention = TimeDistributed(Dense(1, activation='tanh'))(language_model)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(self.encoder_lstm_units)(attention)
        attention = Permute([2, 1])(attention)
        
        applied_attention = Multiply()([language_model, attention])
        language_model = Lambda(lambda xin: K.sum(xin, axis=1))(applied_attention)
        language_model = Lambda(lambda xin: K.expand_dims(xin, axis=-1))(language_model)

        #decoder = CuDNNLSTM(self.decoder_lstm_units, return_sequences=True, kernel_initializer='he_uniform', return_state=True)(language_model)
        #decoder = Dropout(0.1)(decoder)
        # decoder = CuDNNLSTM(self.decoder_lstm_units, return_sequences=True, kernel_initializer='he_uniform', return_state=True)(decoder)
        #decoder = Dropout(0.1)(decoder)
        decoder = CuDNNLSTM(self.decoder_lstm_units, return_sequences=True, kernel_initializer='he_uniform', return_state=False)(language_model)
        output = TimeDistributed(Dense(self.vocab_size, activation=None, kernel_initializer='he_uniform'))(decoder)
        
        model = Model(inputs=[image_input], outputs=output, name=self.name)
        self.model = model

    def predict(self, image):
        '''
        Predict the latex equation given the image

        # Arguments

        image:      a numpy array of shape (H,W,C)

        # Returns
        the string result of the prediction

        '''
        im = np.expand_dims(image, axis=0)
        eq = np.zeros((1, self.embedding_size, self.vocab_size))
        prediction = np.squeeze(self.model.predict([image, eq]))
        return self.decode_lstm(prediction, self.vocab_list)


    
    def decode_lstm(self, model_output, vocab_list):
        '''
        decode the output of the model

        # Arguments

        model_output:       The ouput of predictiction of the model of shape (embedding_size, vocab_size)
        vocab_list:         The array of tokens that the model can predict

        # Returns

        a string of the resulting decoded equation
        '''
        word_ids = np.argmax(model_output, axis=1)
        equation = [vocab_list[x] for x in word_ids]
        equation = ' '.join(equation)
        return equation


    # def save(self):
    #     model_json = self.model.to_json()
    #     with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
    #         json_file.write(model_json)
    #     self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))

    # def load(self, name=""):
    #     output_name = self.name if name == "" else name
    #     with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
    #         loaded_model_json = json_file.read()
    #     self.model = model_from_json(loaded_model_json)
    #     self.model.load_weights("{}/{}.h5".format(self.output_path, output_name))

