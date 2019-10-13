import sys
# we would like to be in the src directory to have access to main files
sys.path.append("..")
import os
import unittest
import numpy as np
from data import create_data


class create_data_Tests(unittest.TestCase):
    def setUp(self):
        self.creator = create_data((256,256), 'output.csv', 'output', 'test_formulas.txt')
   
    def test_single_latex_to_image(self):
        latex_equation = r"\frac{x^2}{2y}"
        image_name = "image.png"
        self.creator.latex_to_img(latex_equation, image_name)
        
        isFileCreated = image_name in os.listdir('.')
        assert isFileCreated == True, "Failed to create a png file in latex_to_image"

        os.system(f'del {image_name}')

    def test_create(self):
        self.creator.create()

        isFileCreated = 'output.csv' in os.listdir('.')
        assert isFileCreated == True, "Failed to create the csv file in create"

        #os.system(f'del output.csv')

if __name__ == "__main__":
    unittest.main()