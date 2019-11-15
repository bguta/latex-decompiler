import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as tv
from imgaug import augmenters as iaa
import torch
from PIL import Image, ImageOps, ImageChops
import io

def aug(scale):
    return iaa.Sequential([
    iaa.Affine(
            scale={"x": scale[0], "y": scale[1]},
            cval=0
        )
])

class generator(Dataset):
    '''
    Generator class to feed the training data

    # Arguments


    # Example

    '''
    def __init__(self,
                list_IDs,
                df,
                base_path,
                shuffle=True):
        
        self.list_IDs       = list_IDs
        self.df             = df
        self.base_path      = base_path
        self.shuffle        = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the length of the data'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index]
        return self.__generate_seq(self.list_IDs[indexes])

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_seq(self, ID):

        im_name = self.df['image_name'].loc[ID]
        image_df = self.df[self.df['image_name'] == im_name]
        img_path = f"{self.base_path}/{im_name}"

        img                     = self.__load_img(img_path)
        encoded_equation        = image_df.Y.values[0]
        encoded_equation_loss   = image_df.Y_loss.values[0]

        X       = img
        Y       = torch.from_numpy(encoded_equation).long()
        Y_Loss  = torch.from_numpy(encoded_equation_loss).long()

        return X, Y, Y_Loss

    def __load_img(self, img_path, transform=True):
        '''
        Load the image as a numpy array
        '''
        img = ImageOps.invert(Image.open(img_path).convert('L'))
        if transform:
            l, u, r, d  = img.getbbox()
            w, h = r-l, d-u
            im_w, im_h = img.size
            w_scale = np.random.uniform(1.0, 1 + (im_w/w - 1)/2)
            h_scale = w_scale * ( w * im_h )/ ( im_w * h )
            #w_scale = h_scale * ( im_w * h ) / ( w * im_h )

            scale = (w_scale, h_scale)
            img = np.array(img)
            if np.random.uniform() > 0.5:
                img =  aug(scale)(image=img)
            else:
                img = iaa.AverageBlur(k=(1,2))(image=img)
        img = tv.to_tensor(img)
        return img