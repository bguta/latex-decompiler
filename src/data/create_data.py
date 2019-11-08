'''
Generator for the image data
'''

import numpy as np
from PIL import Image, ImageOps
import io
from sympy import preview
import pandas as pd
import click
import uuid

class create_data:
    r""" Create a random set of images with a black background and white letters.

    # Arguments

        image_size:     A tuple of the image size eg. (H,W)
        output_csv:     The name of the output csv that contains the names of the images e.g 'latex_imgs.csv'
        output_dir:     Path to the Existing directory to place the images e.g '../data/latex_imgs'
        formual_file:   Text file containing the formulas on every line e.g 'formula.txt'

    # Example

    .. code:: python
        creator = create_data()
        creator.create()
    """

    def __init__(self,
            image_size,
            output_csv,
            output_dir,
            formula_file
            ):
        
        self.__image_size   = image_size
        self.__output_csv   = output_csv
        self.__output_dir   = output_dir
        self.__formula_file = formula_file

    def latex_to_img(self, tex, image_name, image_size=(64,512), background_colour=(255,255,255)):
        '''
        Generate a png image of latex text
    
        # Arguments
    
        tex:                   The tex code to display in the image
        image_name:            The file name i.e 'image'
        image_size:            The image size (H,W) in pixels
        background_colour:     The background colour to use for the image
        '''
        buf = io.BytesIO()
        preview(f'${tex}$', viewer='BytesIO', outputbuffer=buf, euler=False)
        image = Image.open(buf)
        padded_image = self.pad_to_target(image, image_size[0], image_size[1], background_colour)
        ImageOps.grayscale(padded_image).save(image_name)
        #image.save(image_name)
    
    def create(self):
        '''
        Create the directory containing the images as well as the
        corresponding csv which contains

        image_name and latex_equation as headers

        '''
        formulaFile = open(self.__formula_file, 'r', encoding="utf8")
        formulas = formulaFile.read().split('\n')
        formulaFile.close()
        full_len = len(formulas)

        dataset = {'image_name': [], 'latex_equations': []}
        with click.progressbar(range(40000)) as bar:
            for i in bar:
                if len(formulas[i].split(' ')) > 200:
                    continue
                try:
                    im_name = str(uuid.uuid4().hex) + '.png'
                    self.latex_to_img(f"{formulas[i]}", f'{self.__output_dir}/' + im_name, self.__image_size)
                    dataset['image_name'].append(im_name)
                    dataset['latex_equations'].append(f"{formulas[i]}")
                except Exception as e:
                    pass
            pd.DataFrame(data=dataset).to_csv(path_or_buf=self.__output_csv, index=False)
    
    def pad_to_target(self, img, target_height, target_width, background_colour=(255,255,255)):
        '''
        Pad image with 255 to the specified height and width

        This op throws an assertion error if target_height or
        target width is larger than current size

        # Arguments
    
        img:                      The PIL image instance to pad
        target_height:            The integer target height
        target_width:             The integer target width
        background_colour:        The background colour to use for the pad

        # Returns

        The padded PIL image
        '''
        w, h = img.size
        left = top = right = bottom = 0
        pad_image = False
        assert w <= target_width, 'image width is larger than target'
        assert h <= target_height, 'image width is larger than target'

        if target_width > w:
            delta = target_width - w
            left = delta // 2
            right = delta - left
            pad_image = True
        if target_height > h:
            delta = target_height - h
            top = delta // 2
            bottom = delta - top
            pad_image = True
        if pad_image:
            padded_image = ImageOps.expand(img, border=(left, top, right, bottom), fill=background_colour)

        return padded_image
