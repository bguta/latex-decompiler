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
    if abs(w/h - ratio) < 0.5:
        return 0, 0
    pad_h = height_padding
    pad_w = ratio*pad_h + ratio*h - w
    assert pad_w >= 0, 'The padding for the width was too small, please provide an image with a smaller aspect ratio'
    return pad_w, pad_h

def load_img(im_path, im_size=(128,416)):
    '''
    Open the *.png image specified in the im_path and prepocess it
    for our model

    ## Args
    im_path:    the path to the .png file
    im_size:    the size required by our model. default (32,416)

    ## Returns the processed image as a torch tensor
    '''
    img = ImageOps.invert(Image.open(im_path).convert('RGB'))
    pad_dim = cal_shape(*img.size, ratio=im_size[1]//im_size[0], height_padding=1)
    print(f'Padding dimension: {pad_dim}')
    processed_img = tv.pad(img, pad_dim)
    processed_img = tv.resize(processed_img, im_size)
    processed_img = tv.to_tensor(processed_img)
    plt.imshow(processed_img.detach().numpy().transpose(1,2,0))
    plt.show()
    return processed_img.unsqueeze(0)

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
    return input("Please enter the path to the image: ")

def main(model, start_token, vocab_set):
    # parser = argparse.ArgumentParser(description="Latex Decompiler application")
    # parser.add_argument('--img_path', required=True,
    #                     help='path to the stored weights of the model')
    # args = parser.parse_args()
    print("~~ Latex Decompiler application ~~ \n            est. 2019\n")
    while True:
        im_path = prompt()
        img = load_img(im_path)
        encoded_pr = model(img, start_token, 0.0)
        predicted_latex = decode(encoded_pr, vocab_set)
        print(f'Prediction: {predicted_latex}')
        print('')

if __name__ == '__main__':
    stored_model = torch.load('pre_trained/model.pt')
    max_len = 232 + 2

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

