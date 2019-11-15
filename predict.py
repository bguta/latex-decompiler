import torchvision.transforms.functional as tv
from PIL import Image, ImageOps
import argparse
from src.model.model import im2latex
import torch
import numpy as np
import matplotlib.pyplot as plt

def cal_shape(w, h, ratio=13, height_padding=16):
    '''
    Calculate the new shape for the image to maintain the desired aspect ratio.

    ## Args
    h:              the height of the image in pixels
    w:              the width of the image in pixels
    ratio:          the desired aspect ratio, ( new_w / new_h ) default: 13
    height_padding: the padding to use for the height, default: 16

    ## Returns
    (padding_w, padding_h): The padding for each dim to.

    ## Throws asserion error if the padding for the width is negative
    '''
    if abs(w/h - ratio) < 0.1:
        return 0, 0
    pad_h = height_padding
    pad_w = int(ratio*(pad_h + h) - w)
    assert pad_w >= 0, 'The padding for the width was too small, please provide an image with a smaller aspect ratio'
    return pad_w//2, pad_h//2

def load_img(im_path, im_size=(128,416), re=[17/32, 18/32, 19/32]):
    '''
    Open the *.png image specified in the im_path and prepocess it
    for our model

    ## Args
    im_path:    the path to the .png file
    im_size:    the size required by our model. default (128,416)

    ## Returns an array of processed image as a torch tensor
    '''
    imgs = []
    img = ImageOps.invert(Image.open(im_path).convert('L'))
    print(img.getbbox())

    pad_dim = cal_shape(*img.size, ratio=im_size[1]/im_size[0], height_padding=128)
    processed_img = tv.pad(img, pad_dim)
    if pad_dim[0] != 0 or pad_dim[1] != 0:
        for re_ in re:
            re_size = (int(im_size[0]*re_), int(im_size[1]*re_))
            pr_im = tv.resize(processed_img, re_size)
            pd_h = (im_size[0] - re_size[0])//2
            pd_w = (im_size[1] - re_size[1])//2
            imgs.append(tv.pad(pr_im, (pd_w, pd_h)))
    else:
        processed_img = tv.resize(processed_img, im_size)
        imgs.append(processed_img)
    
    imgs = [tv.to_tensor(x).unsqueeze(0) for x in imgs]
    [(plt.imshow(x.squeeze().detach().numpy()), plt.show()) for x in imgs]
    return imgs

def decode(prediction, vocab_list):
    pr = np.squeeze(prediction.detach().numpy())
    token_ids = np.argmax(pr, axis=-1)
    tokens = [vocab_list[x] for x in token_ids]
    decoded_string = ' '.join(tokens)
    decoded_string.strip()
    split = decoded_string.split('<end>')
    processed_tex = split[0].strip()
    return processed_tex

def prompt():
    return input("Please enter the path to the image (enter q to exit): ")

def main(model, start_token, vocab_set):
    print("~~ Latex Decompiler application ~~ \n            est. 2019\n")
    while True:
        im_path = prompt()
        if im_path.lower().strip() == "q":
            print('Exiting')
            break
        try:
            imgs = load_img(im_path)
            pred = []
            for img in imgs:
                encoded_pr = model(img, start_token, -1)
                pred.append(decode(encoded_pr, vocab_set))
            for index, predicted_latex in enumerate(pred):
                print('')
                print(f'Prediction #{index}: {predicted_latex}')
        except AssertionError as e:
            print('Failed to convert image to an input ... please try again')
            pass
        print('')

if __name__ == '__main__':
    stored_model = torch.load('pre_trained/ckpt-33-0.9793.pt')
    max_len = 200 + 2

    vocabFile = open('pre_trained/vocab.txt', 'r', encoding="utf8")
    vocab_tokens = [x.replace('\n', '') for x in vocabFile.readlines()]
    vocabFile.close()
    vocab_size = len(vocab_tokens)

    model = im2latex(vocab_size)
    model.load_state_dict(stored_model['model_state_dict'])
    model.eval()
    
    start_token = np.zeros((1,max_len), dtype=np.int64)
    start_token[0][0] = vocab_size - 2 
    start_token = torch.as_tensor(start_token)

    main(model, start_token, vocab_tokens)

