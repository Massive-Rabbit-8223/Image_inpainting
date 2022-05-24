from matplotlib import image
from utils.image_standardizer import ImageStandardizer
import string
import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def create_dataset(input_dir: string, output_dir: string):
    img_standardizer = ImageStandardizer(input_dir)
    mean, std = img_standardizer.analyze_images()

    print("mean: ", mean)
    print("std: ", std)

    images = []

    for img in img_standardizer.get_standardized_images():

        #print("image shape: ", img.shape)
        #print("image min: ", np.min(img))
        #print("image max: ", np.max(img))
        #print("image mean: ", np.mean(img))

        #plt.imshow(img)
        #plt.show()

        img_resized = torchvision.transforms.Resize((100,100))(torch.swapaxes(torch.swapaxes(torch.tensor(img), 2, 1),1,0))
        
        #plt.imshow(torch.swapaxes(torch.swapaxes(img_resized, 0, 1),1,2))
        #plt.show()

        #print("img_resized shape: ", img_resized.shape)
        #print("img_resized min: ", torch.min(img_resized))
        #print("img_resized max: ", torch.max(img_resized))
        #print("img_resized mean: ", torch.mean(img_resized))

        images.append(img_resized)

    print(len(images))

if __name__ == "__main__":
    input_dir = os.path.abspath(r"Data/Valid_training")
    output_dir = os.path.abspath(r"Data/Dataset")

    print("input_dir: ", input_dir)
    print("output_dir: ", output_dir)

    create_dataset(input_dir, output_dir)