"""
Author: Patrick Mederitsch
Matr.Nr.: K51831808
Exercise 3
"""

from importlib.resources import path
import string
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class ImageStandardizer():
    def __init__(self, input_dir: string) -> None:
        self.files = []
        ### create absolute paths ###
        absolute_input_dir = os.path.abspath(input_dir)
        ### scan input_dir recursively for files ending in .jpg ###
        print("Scan images:")
        for root,d_names,f_names in tqdm(os.walk(absolute_input_dir)):    # go through all folders in directory
            for f in f_names:   # go through all files in current folder
                image_file_path = os.path.join(root, f)
                _, file_ending = os.path.splitext(image_file_path)
                if file_ending == ".jpg":   # check for file ending
                    self.files.append(os.path.abspath(image_file_path))
        if len(self.files) == 0:    # if no files have been added to the list, it means no files with the right file ending were found
            raise ValueError("No files with ending .jpg found!")
        else:
            self.files = sorted(self.files) # store sorted absolute file paths
        
        self.mean = None
        self.std = None

    def analyze_images(self) -> tuple:
        """
        Compute mean and standard deviation for each color channel of all images in self.files.
        return (self.mean, self.std)
        self.mean -> shape=(3,), dtype=np.float64
        self.std -> shape=(3,), dtype=np.float64
        """
        self.mean = np.zeros((3,), dtype=np.float64)
        self.std = np.zeros((3,), dtype=np.float64)

        mean = []
        std = []

        print("Analyze images:")
        for image_file_path in tqdm(self.files[:5000]):
            image = np.array(Image.open(image_file_path), dtype=np.float64) # load image as array
            mean.append(np.mean(image, axis=(0,1), dtype=np.float64))   # calculate the mean for all channels
            std.append(np.std(image, axis=(0,1), dtype=np.float64))     # calculate the std for all channels

        self.mean = np.mean(np.array(mean, dtype=np.float64), dtype=np.float64, axis=0) # take the average of all means
        self.std = np.mean(np.array(std, dtype=np.float64), dtype=np.float64, axis=0)   # take the average of all stds

        return (self.mean, self.std)

    def get_standardized_images(self) -> np.ndarray:
        """
        Open images as array.
        Standardize images by substracting self.mean and dividing by self.std
        Yield standardized images.
        """
        if (self.mean.all() == None) or (self.std.all() == None):   # check if mean and std values are valid
            raise ValueError("Mean and standard deviation can not have a value of None!")

        for image_file_path in tqdm(self.files[:5000]):
            image = np.array(Image.open(image_file_path), dtype=np.float32) # load image as array
            standardized_image = (image-self.mean) / self.std   # standardize image to have zero mean and unit variance
            yield standardized_image.astype('float32')