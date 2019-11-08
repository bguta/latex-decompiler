import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image, ImageOps
import io
from sympy import preview

def encode_equation(string, vocab_list, dim, is_for_loss=False):
    encoding = [vocab_list.index(x) if x in vocab_list else 0 for x in string.split(' ')]
    if not is_for_loss:
        encoding.insert(0, len(vocab_list) - 2) # insert the start token
    encoding += [len(vocab_list) - 1] # insert the end token
    encoding += [0]*(dim - len(encoding)) # pad the rest
    return np.array(encoding)


class generator(Dataset):
    '''
    Generator class to feed the training data

    # Arguments


    # Example

    '''
    def __init__(self,
                list_IDs,
                df,
                dim,
                eq_dim,
                batch_size,
                base_path,
                vocab_list,
                shuffle=True,
                n_channels=3):
        
        self.list_IDs       = list_IDs
        self.df             = df
        self.dim            = dim
        self.eq_dim         = eq_dim
        self.batch_size     = batch_size
        self.base_path      = base_path
        self.vocab_list     = vocab_list
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
        return self.__generate_seq(list_IDs_batch)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_seq(self, list_IDs_batch):
        X         = np.empty((self.batch_size, self.n_channels, *self.dim), dtype=np.float32)
        Y         = np.zeros((self.batch_size, self.eq_dim), dtype=np.int64)
        Y_forLoss = np.zeros((self.batch_size, self.eq_dim), dtype=np.int64)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['image_name'].loc[ID]
            image_df = self.df[self.df['image_name'] == im_name]
            img_path = f"{self.base_path}/{im_name}"


            img = self.__load_img(img_path)
            raw_equation = image_df.latex_equations.values[0]
            encoded_equation = encode_equation(raw_equation, self.vocab_list, self.eq_dim)
            encoded_equation_loss = encode_equation(raw_equation, self.vocab_list, self.eq_dim, is_for_loss=True)
 

            X[i, ]    = img
            Y[i, ]    = encoded_equation
            Y_forLoss[i, ] = encoded_equation_loss

        return [torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(Y_forLoss)]

    def __load_img(self, img_path, transform=True):
        '''
        Load the image as a numpy array
        '''
        img = ImageOps.invert(Image.open(img_path).convert('L'))
        img = np.expand_dims(np.array(img), axis=-1)
        img = np.expand_dims(img.transpose((2,0,1)), axis=0)/255.
        return img