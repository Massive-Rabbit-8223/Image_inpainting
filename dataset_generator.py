from matplotlib import image
from utils.image_standardizer import ImageStandardizer
from utils.feature_target_creator import ex4
import string
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import splitfolders
from tqdm import tqdm
import pickle
 

class ImageDataset(Dataset):
    def __init__(self, train_std, train_mean, img_standardizer: ImageStandardizer, spacing_range: tuple, offset_range: tuple) -> None:
        features = []
        targets = []
        images = []
        known_pixels = []
        spacing_list = []
        offset_list = []
        print("Create Dataset:")
        im_shape = 100
        resize_transforms = transforms.Compose([
            transforms.Resize(size=im_shape),
            transforms.CenterCrop(size=(im_shape, im_shape)),
        ])
        for img, gt in img_standardizer.get_normalized_images():

            img_resized = resize_transforms(torch.from_numpy(np.moveaxis(img, -1, 0)))

            spacing_x = np.random.randint(spacing_range[0], spacing_range[1]+1)
            spacing_y = np.random.randint(spacing_range[0], spacing_range[1]+1)
            offset_x = np.random.randint(offset_range[0], offset_range[1]+1)
            offset_y = np.random.randint(offset_range[0], offset_range[1]+1)

            input_array, known_array, target_array = ex4(
                image_array=torch.moveaxis(img_resized.clone().detach(), 0, -1).cpu().detach().numpy(), 
                offset=(offset_x, offset_y), 
                spacing=(spacing_x, spacing_y)
            )

            features.append(torch.from_numpy(input_array))
            targets.append(torch.from_numpy(target_array))
            images.append(img_resized)
            known_pixels.append(torch.from_numpy(known_array))
            spacing_list.append((spacing_x, spacing_y))
            offset_list.append((offset_x, offset_y))

        self.length = len(features)

        self.features = features
        self.targets = targets
        self.images = images
        self.known_pixels = known_pixels
        self.spacing = spacing_list
        self.offset = offset_list
        self.mean = train_mean
        self.std = train_std

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.features[index], self.images[index], self.known_pixels[index], self.spacing[index], self.offset[index])

class TestDataset(Dataset):
    def __init__(self, train_std, train_mean, path_to_test_pkl) -> None:
        with open(path_to_test_pkl, 'rb') as f:
            test_set_dict = pickle.load(f)

        self.length = len(test_set_dict['input_arrays'])

        self.features = test_set_dict['input_arrays']
        self.known_arrays = test_set_dict['known_arrays']
        self.spacings = test_set_dict['spacings']
        self.offsets = test_set_dict['offsets']

        self.train_std = train_std
        self.train_mean = train_mean

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        feature_normalized = self.normalize_image(self.features[index].copy())
        return (feature_normalized, self.known_arrays[index], self.spacings[index], self.offsets[index])

    def normalize_image(self, image):
        return image / 255

    def denormalize_image(self, image):
        return image * 255


def create_dataset(input_dirs: tuple, spacing_range: tuple, offset_range: tuple) -> tuple:
    train_dir = input_dirs[0]
    val_dir = input_dirs[1]
    test_dir = input_dirs[2]

    train_img_standardizer = ImageStandardizer(train_dir)
    train_mean, train_std = train_img_standardizer.analyze_images() # take normalizing constants from training data

    val_img_standardizer = ImageStandardizer(val_dir)
    val_img_standardizer.mean = train_mean  # use normalizing constants from training set and apply on validation set
    val_img_standardizer.std = train_std

    train_dataset = ImageDataset(
        train_std=train_std, 
        train_mean=train_mean, 
        img_standardizer=train_img_standardizer, 
        spacing_range=spacing_range, 
        offset_range=offset_range
    )
    val_dataset = ImageDataset(
        train_std=train_std, 
        train_mean=train_mean, 
        img_standardizer=val_img_standardizer, 
        spacing_range=spacing_range, 
        offset_range=offset_range
    )
    ## create test dataset
    test_dataset = TestDataset(
        train_std=train_std, 
        train_mean=train_mean, 
        path_to_test_pkl=test_dir
    )

    return (train_dataset, val_dataset, test_dataset)


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
