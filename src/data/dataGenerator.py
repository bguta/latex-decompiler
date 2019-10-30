import warnings
warnings.filterwarnings("ignore") # ignore tensorflow warnings

import numpy as np
import tensorflow.keras as keras
from PIL import Image, ImageOps
import io
from sympy import preview
from albumentations import CropNonEmptyMaskIfExists, Compose

def get_augmentations(p=1.0, h=128, w=1024):
    augmentations = Compose([
        CropNonEmptyMaskIfExists(height=h, width=w, p=1.0),
    ], p=p)

    return augmentations

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

def encode_ctc_equation(string, vocab_list):
    encoding = [vocab_list.index(x) if x in vocab_list else 0 for x in string.split(' ')]
    encoding.insert(0, len(vocab_list) - 2) # insert the start token
    encoding += [len(vocab_list) - 1] # insert the end token
    #encoding += [0]*(dim[0] - len(encoding))
    return np.array(encoding, dtype=np.float32)

def crop_im(im, padding=0.1):
    '''
    Take in a black and white image and crop to region
    contating white (1.0) and pad with % padding
    '''
    rows, cols = im.shape[:-1]
    non_empty_columns = np.where(im.max(axis=0)>0)[0]
    non_empty_rows = np.where(im.max(axis=1)>0)[0]
    if len(non_empty_columns) == 0 or len(non_empty_rows) == 0:
        return im
    cropBox = (min(non_empty_rows * (1 - padding)), min(max(non_empty_rows * (1 + padding)), rows),
                min(non_empty_columns * (1 - padding)), min(max(non_empty_columns * (1 + padding)), cols))
    cropped = im[int(cropBox[0]):int(cropBox[1])+1, int(cropBox[2]):int(cropBox[3])+1 , :]
    return cropped

class generator(keras.utils.Sequence):
    '''
    Generator class to feed the training data

    # Arguments


    # Example

    '''
    def __init__(self,
                list_IDs,
                df,
                target_df,
                dim,
                eq_dim,
                batch_size,
                base_path,
                preprocess,
                vocab_list,
                shuffle=True,
                n_channels=1):
        
        self.list_IDs       = list_IDs
        self.df             = df
        self.target_df      = target_df
        self.dim            = dim
        self.eq_dim         = eq_dim
        self.batch_size     = batch_size
        self.base_path      = base_path
        self.vocab_list     = vocab_list
        self.preprocess     = preprocess
        self.shuffle        = shuffle
        self.n_channels     = n_channels
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        return self.__generate_token_seq(list_IDs_batch)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_seq(self, list_IDs_batch):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.eq_dim), dtype=np.float32)
        #Y = np.empty((self.batch_size, self.eq_dim[0]), dtype=np.float32)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['image_name'].loc[ID]
            image_df = self.target_df[self.target_df['image_name'] == im_name]
            img_path = f"{self.base_path}/{im_name}"


            img = self.__load_grayscale(img_path)
            raw_equation = image_df.latex_equations.values[0]
            encoded_equation = encode_equation(raw_equation, self.vocab_list, self.eq_dim)

            img = self.preprocess(img)

            X[i, ] = np.expand_dims(img, axis=-1)
            Y[i, ] = encoded_equation

        return X, Y
    
    def __generate_token_seq(self, list_IDs_batch):
        X = []
        Y = []

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['image_name'].loc[ID]
            image_df = self.target_df[self.target_df['image_name'] == im_name]
            img_path = f"{self.base_path}/{im_name}"


            img = self.__load_grayscale(img_path)
            img = self.preprocess(img)
            img = np.expand_dims(img, axis=-1)
            #img = crop_im(img)
            raw_equation = image_df.latex_equations.values[0]
            encoded_equation = encode_ctc_equation(raw_equation, self.vocab_list)
            encoded_equation = encoded_equation.reshape(len(encoded_equation),1)
            #time_seq_equations = self.__make_time_seq(encoded_equation, self.eq_dim[0])
            img = np.expand_dims(img, axis=0)
            #img = self.preprocess(img)
            #img = np.repeat(img, 3, axis=-1)

            #for time_step in range(len(encoded_equation) - 1):
            batch = len(encoded_equation)-1
            X = [np.repeat(img, batch, axis=0), encoded_equation[:-1,:]]
            
            Y = encoded_equation[1:,:]
        return X, Y


    def __make_time_seq(self, y, dim):
        '''
        y must be a rank 1 tensor
        '''
        length = len(y)
        matrix = np.zeros((length, dim)).astype(np.float32)
        valid_indices = np.nonzero(y)[0][-1] + 1
        for i in range(valid_indices):
            matrix[i][:i+1] = y[:i+1]
        return matrix

    def __load_grayscale(self, img_path):
        '''
        Load the image as a numpy array
        '''
        img = ImageOps.invert(Image.open(img_path).convert('L'))
        image = np.array(img, dtype=np.float32)
        return image

    def __load_rgb(self, img_path):
        '''
        Load the image as a numpy array
        '''
        img = ImageOps.invert(Image.open(img_path).convert('RGB'))
        image = np.array(img, dtype=np.float32)
        return image