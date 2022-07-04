from matplotlib import image
from utils.image_standardizer import ImageStandardizer
from utils.feature_target_creator import ex4
import string
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
import splitfolders
from tqdm import tqdm
 

class ImageDataset(Dataset):
    def __init__(self, img_standardizer: ImageStandardizer, spacing_range: tuple, offset_range: tuple) -> None:
        features = []
        targets = []
        images = []
        known_pixels = []
        spacing_list = []
        offset_list = []
        print("Create Dataset:")
        for img in img_standardizer.get_standardized_images():

            img_resized = torchvision.transforms.Resize((100,100))(torch.swapaxes(torch.swapaxes(torch.tensor(img), 2, 1),1,0)) # resized to (100, 100) because test images are in that shape
            
            spacing_x = np.random.randint(spacing_range[0], spacing_range[1]+1)
            spacing_y = np.random.randint(spacing_range[0], spacing_range[1]+1)
            offset_x = np.random.randint(offset_range[0], offset_range[1]+1)
            offset_y = np.random.randint(offset_range[0], offset_range[1]+1)

            input_array, known_array, target_array = ex4(
                image_array=torch.swapaxes(torch.swapaxes(img_resized, 0, 1),1,2).cpu().detach().numpy(), 
                offset=(offset_x, offset_y), 
                spacing=(spacing_x, spacing_y)
            )  # needs to be variable

            features.append(torch.tensor(input_array))
            targets.append(torch.tensor(target_array))
            images.append(img_resized)
            known_pixels.append(torch.tensor(known_array))
            spacing_list.append((spacing_x, spacing_y))
            offset_list.append((offset_x, offset_y))

        self.length = len(features)

        self.features = features
        self.targets = targets
        self.images = images
        self.known_pixels = known_pixels
        self.spacing = spacing_list
        self.offset = offset_list

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.features[index], self.targets[index], self.images[index], self.known_pixels[index], self.spacing[index], self.offset[index])

def create_dataset(input_dirs: tuple, spacing_range: tuple, offset_range: tuple) -> tuple:
    train_dir = input_dirs[0]
    val_dir = input_dirs[1]

    train_img_standardizer = ImageStandardizer(train_dir)
    train_mean, train_std = train_img_standardizer.analyze_images() # take normalizing constants from training data

    val_img_standardizer = ImageStandardizer(val_dir)
    val_img_standardizer.mean = train_mean  # use normalizing constants from training set and apply on validation set
    val_img_standardizer.std = train_std

    train_dataset = ImageDataset(train_img_standardizer, spacing_range, offset_range)
    val_dataset = ImageDataset(val_img_standardizer, spacing_range, offset_range)
    ## create test dataset

    return (train_dataset, val_dataset)


if __name__ == "__main__":
    input_dir = os.path.abspath(r"../../Data")
    output_dir = os.path.abspath(r"../../Data/Dataset")

    splitfolders.ratio(input_dir, output=output_dir, seed=1337, ratio=(.8, 0.2))
    
    train_input_dir = os.path.abspath(r"../../Data/Dataset/train")
    val_input_dir = os.path.abspath(r"../../Data/Dataset/val")
    output_dir = os.path.abspath(r"../../Data/Dataset")

    print("train_input_dir: ", train_input_dir)
    print("val_input_dir: ", val_input_dir)

    train_dataset, val_dataset = create_dataset((train_input_dir, val_input_dir))

    print("train_dataset: ", len(train_dataset))
    print("val_dataset: ", len(val_dataset))
