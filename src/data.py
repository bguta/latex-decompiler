'''
Generator for the image data
'''

import numpy as np
from PIL import Image, ImageOps
import io
import matplotlib.pyplot as plt
import pandas as pd

class create_data:
    r""" Create a random set of images with a black background and white letters.

    # Arguments

        image_size:     A tuple of the image size eg. (256,256)
        output_csv:     The name of the output csv that contains the names of the images e.g 'latex_imgs.csv'
        output_dir:     Path to the directory to place the images e.g '../data/latex_imgs'
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

    def latex_to_img(self, tex, image_name, image_size=(256,256), offset=10, fontsize=40, background_colour=(255,255,255)):
        '''
        Generate a png image of latex text
    
        # Arguments
    
        tex:                   The tex code to display in the image
        image_name:            The file name i.e 'image'
        image_size:            The image size (width,height) in inches at 100 dpi
        offset:                The offest of the text in the image
        fontsize:              The font size in points
        background_colour:     The background colour to use for the image
        '''
        buf = io.BytesIO()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.axis('off')
        plt.text(0.0, 0.5, f'${tex}$', fontsize=fontsize, ha='left', va='center')
        plt.savefig(buf, format='png', bbox_inches="tight")
        plt.close()
        image = Image.open(buf)
        ImageOps.grayscale(image).save(image_name)
        #image.save(image_name)
    
    def create(self):
        '''
        Create the directory containing the images as well as the
        corresponding csv which contains

        image_name and latex_equation as headers

        '''
        formulaFile = open(self.__formula_file, 'r')
        formulas = formulaFile.read().split("\n")
        formulaFile.close()

        dataset = {'image_name': [], 'latex_equations': []}

        for i in range(len(formulas)):
            im_name = f'image_{i}.png'
            dataset['image_name'].append(im_name)
            dataset['latex_equations'].append(formulas[i])
        pd.DataFrame(data=dataset).to_csv(path_or_buf=self.__output_csv, index=False)
