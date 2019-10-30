import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings
import sys
# we would like to be in the src directory to have access to main files
sys.path.append("..")


from tensorflow.keras.layers import CuDNNLSTM, CuDNNGRU, Input, Embedding, Dense, RepeatVector, TimeDistributed, Reshape, concatenate, Dropout, Multiply, Lambda, Add, Activation, Flatten
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from model.cnn import cnn
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf

class im2latex:
    r""" Create the image to latex converter model.

    # Arguments

        encoder_lstm_units:     The dimensionality of the output space for encoder LSTM layers
        decoder_lstm_units:     The dimensionality of the output space for decoder LSTM layers
        vocab_list:             The array of possible outputs of the language model


    # Example

    .. code:: python
    latex_model = im2latx(decoder_lstm_units, vocab_list)

    model = latex_model.model
    """

    def __init__(self, encoder_lstm_units, decoder_lstm_units, vocab_list, batch_size=1, visualize=False):
        self.name = 'im2latex'
        self.encoder_lstm_units = encoder_lstm_units
        self.decoder_lstm_units = decoder_lstm_units
        self.vocab_list = vocab_list
        self.vocab_size = len(self.vocab_list)
        self.batch_size = batch_size

        # encoder
        image_input = Input(batch_shape=(batch_size,32,416,1), name='image_input')
        feature_extractor = cnn(image_input) #tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=image_input)

        #features = Lambda(lambda xin: K.expand_dims(xin, axis=-1))(feature_extractor) #cnn(image_input)
        features = Reshape((8*26,512), name='reshape_features')(feature_extractor)
        features = CuDNNGRU(self.decoder_lstm_units, return_sequences=True, return_state=False, stateful=False)(features)
        #features = Dense(self.encoder_lstm_units, activation='relu', name='embed_features')(features)

        # lstm_input = Input(shape=(661,), name='previous_output_seq')
        # word_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.decoder_lstm_units, name='embed_previous')(lstm_input)

        last_token = Input(batch_shape=(batch_size,1), name='previous_output_token')
        token = Embedding(input_dim=self.vocab_size, output_dim=self.encoder_lstm_units, name='embed_previous_token')(last_token)
        
        # encode_word, state_h, state_c = CuDNNLSTM(self.decoder_lstm_units, return_sequences=False, kernel_initializer='he_uniform', return_state=True, name='encode_last_state')(word_embedding)
        # encode_word = RepeatVector(1, name='add_time_dim')(encode_word)
        #hidden_state = [state_h, state_c]

        
        #full_hstate = Repeat(1)(concatenate([h1, h2], axis=1,  name='combine_hidden_states'))
        full_h_dense = Dense(self.encoder_lstm_units, name='densify_token_forAttention')(token)
        feature_dense = Dense(self.encoder_lstm_units, name='densify_features_forAttention')(features)

        attention = Add(name='add_densified')([feature_dense, full_h_dense])
        #attention = Dropout(0.5)(attention)
        attention = Lambda(lambda xin: K.tanh(xin))(attention)
        attention = Dense(1, name='make_attention_weights')(attention)
        attention = Lambda(lambda xin: K.softmax(xin, axis=1))(attention)
        #attention = Dropout(0.5)(attention)
        context_vector = Multiply(name='apply_weights')([features, attention])
        context_vector = Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(context_vector)
        context_vector = Lambda(lambda xin: tf.expand_dims(xin, axis=1), name='add_time_dim_toContextVec')(context_vector) # RepeatVector(1, name='add_time_dim_toContextVec')(context_vector)

        #combined = concatenate([context_vector, token], axis=-1, name='combine_context_and_word')
        #combined = Dropout(0.3)(combined)

        #decoder, hidden1, hidden2 = CuDNNLSTM(self.decoder_lstm_units, return_sequences=False, kernel_initializer='he_uniform', return_state=True, name='decode_context_and_word')(combined)
        #decoder = RepeatVector(1, name='add_time_dim_for_decoder')(decoder)
        decoder, h1 = CuDNNGRU(self.decoder_lstm_units, return_sequences=False, return_state=True, stateful=True)(context_vector)
        output = Dense(self.vocab_size, activation=None, name='next_token')(decoder)
        
        if visualize:
            model = Model(inputs=[image_input, last_token], outputs=[output, attention], name=self.name)
        else:
            model = Model(inputs=[image_input, last_token], outputs=[output], name=self.name)
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

