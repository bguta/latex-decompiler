import sys
# we would like to be in the src directory to have access to main files
sys.path.append("..")
import os
import unittest
import numpy as np
from create_data import create_data

def delete_temp_file(file_name):
    '''
    delete a file created by the test

    use os.system(f'rm -rf {file_name}') if not on windows
    '''
    os.system(f'del /q {file_name}')


class create_data_Tests(unittest.TestCase):
    def setUp(self):
        self.creator = create_data((32,512), 'output.csv', 'output', 'test_formulas.txt')
   
    def test_single_latex_to_image(self):
        latex_equation = r"\frac{x^2}{2y}"
        image_name = "image.png"
        self.creator.latex_to_img(latex_equation, image_name)
        
        isFileCreated = image_name in os.listdir('.')
        assert isFileCreated == True, "Failed to create a png file in latex_to_image"

        delete_temp_file(image_name)

    def test_create(self):
        self.creator.create()

        isFileCreated = 'output.csv' in os.listdir('.')
        assert isFileCreated == True, "Failed to create the csv file in create"

        delete_temp_file('output.csv')
        delete_temp_file('output')

if __name__ == "__main__":
    unittest.main()