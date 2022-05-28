from pickle import TRUE
import os
from pickletools import optimize
import splitfolders
import torch

from utils.validate import validate_images
from dataset_generator import create_dataset
from models import Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

path_to_experiments = os.path.abspath(r"../../ImageInpaintingExperiments")
if not os.path.exists(path_to_experiments):
    os.makedirs(path_to_experiments)

experiment_id = "Test01"
path_to_current_experiment = os.path.join(path_to_experiments, experiment_id)
if not os.path.exists(path_to_current_experiment):
    os.makedirs(path_to_current_experiment)

config = {
    "Data": {
        "raw_img_dir": os.path.join(path_to_experiments, "training/"),
        "data_dir": os.path.join(path_to_current_experiment, "Data/"),
        "valid_img_folder": r"Valid_data",
        "log_file": "logfile.log",
        "formatter": "06d",
        "train_val_split_folder": r"Dataset",
        "seed": 1337,
        "train_val_split_ratio": (0.8, 0.2)
    },
    "Model": {
        "lr": 1e-3,
        "path_to_model": os.path.join(path_to_current_experiment, "model.pth")
    },
    "Training": {
        "epochs": 100
    }
}

FILTER_IMAGES = True
CREATE_DATASET = True
TRAIN_MODEL = False
LOG_MODEL = False

train_input_dir = os.path.join(os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), "train/")
val_input_dir = os.path.join(os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), "val/")


if CREATE_DATASET == True:
    if FILTER_IMAGES == True:
        print("Filter images:")
        print("input dir: ", config["Data"]["raw_img_dir"])
        valid_img_count = validate_images(
            input_dir=config["Data"]["raw_img_dir"], 
            output_dir=os.path.join(config["Data"]["data_dir"], config["Data"]["valid_img_folder"]), 
            log_file=os.path.join(config["Data"]["data_dir"], config["Data"]["log_file"]), 
            formatter=config["Data"]["formatter"]
        )
        print(f"Number of valid images: {valid_img_count}")

    print("Create dataset:")

    splitfolders.ratio(
        input=config["Data"]["data_dir"], 
        output=os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), 
        seed=config["Data"]["seed"], 
        ratio=config["Data"]["train_val_split_ratio"]
    )
    
    print("train_input_dir: ", train_input_dir)
    print("val_input_dir: ", val_input_dir)

    train_dataset, val_dataset = create_dataset(
        (train_input_dir, val_input_dir)
    )

    print(f"train_dataset length: {len(train_dataset)}")
    print(f"val_dataset length: {len(val_dataset)}")

    print("Saving datasets...")
    torch.save(train_dataset, os.path.join(path_to_current_experiment, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(path_to_current_experiment, "val_dataset.pt"))
    print("Successfully saved datasets!")
else:
    print("Loading dataset:")
    train_dataset = torch.load(os.path.join(path_to_current_experiment, "train_dataset.pt"))
    val_dataset = torch.load(os.path.join(path_to_current_experiment, "val_dataset.pt"))

    print(f"train_dataset length: {len(train_dataset)}")
    print(f"val_dataset length: {len(val_dataset)}")

train_dataloader = None
val_dataloader = None

model = Net()
optimizer = optim.Adam(model.parameters(), lr=config["Model"]["lr"])
criterion = nn.MSELoss()

if TRAIN_MODEL == True:
    model.train_network(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config["Training"]["epochs"],
        trainloader=train_dataloader
    )

    torch.save(model.state_dict(), config["Model"]["path_to_model"])


