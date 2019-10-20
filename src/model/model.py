import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import sys
# we would like to be in the src directory to have access to main files
sys.path.append("..")


from tensorflow.keras.layers import CuDNNLSTM, Input, Embedding, Dense, RepeatVector, TimeDistributed, Reshape, concatenate, Dropout, Multiply, Lambda, Add, Activation
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from model.cnn import cnn
import numpy as np
import tensorflow.keras.backend as K

class im2latex:
    r""" Create the image to latex converter model.

    # Arguments

        decoder_lstm_units:     The dimensionality of the output space for decoder LSTM layers
        vocab_list:             The array of possible outputs of the language model


    # Example

    .. code:: python
    latex_model = im2latx(decoder_lstm_units, vocab_list)

    model = latex_model.model
    """

    def __init__(self, decoder_lstm_units, vocab_list, batch_size=16):
        self.name = 'im2latex'
        self.decoder_lstm_units = decoder_lstm_units
        self.vocab_list = vocab_list
        self.vocab_size = len(self.vocab_list)
        self.batch_size = batch_size

        # encoder
        image_input = Input(shape=(128,1024,1), name='image_input')
        features = cnn(image_input)
        features = Reshape((64*8, 256), name='reshape_features')(features)
        features = Dense(self.decoder_lstm_units, activation='relu', name='embed_features')(features)

        lstm_input = Input(shape=(661,), name='previous_output')
        word_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.decoder_lstm_units, name='embed_previous')(lstm_input)
        
        encode_word, state_h, state_c = CuDNNLSTM(self.decoder_lstm_units, return_sequences=False, kernel_initializer='he_uniform', return_state=True, name='encode_last_state')(word_embedding)
        encode_word = RepeatVector(1, name='add_time_dim')(encode_word)
        #h1_in = Input(shape=(self.decoder_lstm_units,), name='h1_in')
        #h2_in = Input(shape=(self.decoder_lstm_units,), name='h2_in')
        #hidden_state = [state_h, state_c]
        
        #full_hstate = RepeatVector(1, name='combine_hidden_states')(concatenate([h1_in, h2_in], axis=1))
        full_h_dense = Dense(self.decoder_lstm_units, name='densify_states_forAttention')(encode_word)
        feature_dense = Dense(self.decoder_lstm_units, name='densify_features_forAttention')(features)

        attention = Add(name='add_densified')([feature_dense, full_h_dense])
        attention = Dropout(0.5)(attention)
        attention = Activation('tanh')(attention)
        attention = Dense(1, activation='softmax', name='make_attention_weights')(attention)
        attention = Dropout(0.5)(attention)
        context_vector = Multiply(name='apply_weights')([features, attention])
        context_vector = Lambda(lambda xin: K.sum(xin, axis=1))(context_vector)
        context_vector = RepeatVector(1, name='add_time_dim_toContextVec')(context_vector)

        combined = concatenate([context_vector, encode_word], axis=1, name='combine_context_and_word')

        #decoder, hidden1, hidden2 = CuDNNLSTM(self.decoder_lstm_units, return_sequences=False, kernel_initializer='he_uniform', return_state=True, name='decode_context_and_word')(combined)
        #decoder = RepeatVector(1, name='add_time_dim_for_decoder')(decoder)
        decoder, h1, h2 = CuDNNLSTM(self.decoder_lstm_units, return_sequences=False, kernel_initializer='he_uniform', return_state=True, stateful=False)(combined, initial_state=[state_h, state_c])
        output = Dense(self.vocab_size, activation=None, kernel_initializer='he_uniform', name='equation')(decoder)
        
        model = Model(inputs=[image_input, lstm_input], outputs=[output], name=self.name)
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

