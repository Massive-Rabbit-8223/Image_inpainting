import pickle
from pickle import TRUE
import os
from pickletools import optimize
from pyexpat import model
import splitfolders
import json
import h5py

from utils.validate import validate_images
from dataset_generator import create_dataset
from models import Autoencoder, SimpleCNN, train_network, create_data_dict, create_test_predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

path_to_experiments = os.path.abspath(r"../../ImageInpaintingExperiments")
if not os.path.exists(path_to_experiments):
    os.makedirs(path_to_experiments)

experiment_id = "Submission_11"
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
        "train_val_split_ratio": (0.99, 0.01),
        "spacing_range": (2, 6),
        "offset_range": (1, 8)
    },
    "Model": {
        "lr": 1e-3,
        "path_to_model": os.path.join(path_to_current_experiment, "model.pth"),
        "n_hidden_layers": 6,
        "n_kernels": 64,
        "kernel_size": 5
    },
    "Training": {
        "epochs": 200,
        "batch_size": 400
    }
}

FILTER_IMAGES = False
CREATE_DATASET = False
TRAIN_MODEL = True
CREATE_DATA_DICTS = True
CREATE_TEST_SUBMISSION = True

train_input_dir = os.path.join(os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), "train/")
val_input_dir = os.path.join(os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), "val/")
test_input_dir = os.path.join(path_to_experiments, "test/inputs.pkl")
path_to_train_data_dict = path_to_current_experiment + "/train_data_dict.hdf5"
path_to_val_data_dict = path_to_current_experiment + "/val_data_dict.hdf5"

if CREATE_DATASET == True:
    if FILTER_IMAGES == True:
        print("Filter images:")
        print("input dir: ", config["Data"]["raw_img_dir"])

        ## Check if images are in correct format etc. ##
        valid_img_count = validate_images(
            input_dir=config["Data"]["raw_img_dir"], 
            output_dir=os.path.join(config["Data"]["data_dir"], config["Data"]["valid_img_folder"]), 
            log_file=os.path.join(config["Data"]["data_dir"], config["Data"]["log_file"]), 
            formatter=config["Data"]["formatter"]
        )
        print(f"Number of valid images: {valid_img_count}")

    print("Create dataset:")

    ## Split data into train an validation set ##
    splitfolders.ratio(
        input=config["Data"]["data_dir"], 
        output=os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), 
        seed=config["Data"]["seed"], 
        ratio=config["Data"]["train_val_split_ratio"]
    )
    
    print("train_input_dir: ", train_input_dir)
    print("val_input_dir: ", val_input_dir)

    ## create datasets for training and validation ##
    train_dataset, val_dataset, test_dataset = create_dataset(
        input_dirs=(train_input_dir, val_input_dir, test_input_dir),
        spacing_range=config["Data"]["spacing_range"],
        offset_range=config["Data"]["offset_range"]
    )

    print(f"train_dataset length: {len(train_dataset)}")
    print(f"val_dataset length: {len(val_dataset)}")
    print(f"test_dataset length: {len(test_dataset)}")

    print("Saving datasets...")
    torch.save(train_dataset, os.path.join(path_to_current_experiment, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(path_to_current_experiment, "val_dataset.pt"))
    torch.save(test_dataset, os.path.join(path_to_current_experiment, "test_dataset.pt"))
    print("Successfully saved datasets!")
else:
    print("Loading dataset:")
    train_dataset = torch.load(os.path.join(path_to_current_experiment, "train_dataset.pt"))
    val_dataset = torch.load(os.path.join(path_to_current_experiment, "val_dataset.pt"))
    test_dataset = torch.load(os.path.join(path_to_current_experiment, "test_dataset.pt"))
    print(f"train_dataset length: {len(train_dataset)}")
    print(f"val_dataset length: {len(val_dataset)}")
    print(f"test_dataset length: {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=config["Training"]["batch_size"], shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["Training"]["batch_size"], shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print(f'Using {device} device')

#model = Autoencoder()
model = SimpleCNN(
    n_hidden_layers=config["Model"]["n_hidden_layers"],
    n_kernels=config["Model"]["n_kernels"],
    kernel_size=config["Model"]["kernel_size"]
)
optimizer = optim.Adam(model.parameters(), lr=config["Model"]["lr"])
criterion = nn.MSELoss()
criterion_individual = nn.MSELoss(reduction='none')
model.to(device)

if TRAIN_MODEL == True:
    train_network(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        n_epochs=config["Training"]["epochs"],
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        experiment_id=experiment_id
    )

    torch.save(model.state_dict(), config["Model"]["path_to_model"])
else:
    model.load_state_dict(torch.load(config["Model"]["path_to_model"]))

## File handling Utils ##

def save_dict_hdf5(dict_to_save, file_name):
    with h5py.File(file_name, 'w') as f:
        for key, value in dict_to_save.items():
            f.create_dataset(key, data=value)

def load_dict_hdf5(file_name):
    dictionary = {}
    with h5py.File(file_name, 'r') as f:
        for key in f.keys():
            dictionary[key] = f[key][:]
    
    return dictionary

def save_config_json(config_dict, file_name="config.json"):
    with open(os.path.join(path_to_current_experiment, file_name), 'w') as f:
        json.dump(config_dict, f, indent = 4)

save_config_json(config)

if CREATE_DATA_DICTS == True:
    print("Creating Training Data Dict...")
    train_data_dict = create_data_dict(
        model=model, 
        criterion=criterion_individual,
        dataloader=train_dataloader,
        device=device
    )
    print("Saving Training Data Dict...")
    save_dict_hdf5(train_data_dict, path_to_train_data_dict)

    print("Creating Val Data Dict...")
    val_data_dict = create_data_dict(
        model=model, 
        criterion=criterion_individual,
        dataloader=val_dataloader,
        device=device
    )
    print("Saving Val Data Dict...")
    save_dict_hdf5(val_data_dict, path_to_val_data_dict)

if CREATE_TEST_SUBMISSION == True:
    # call 'create_test_predictions' function
    predictions_list = create_test_predictions(
        model=model, 
        device=device,
        test_dataset=test_dataset
    )
    # save prediction to pickle file
    with open(os.path.join(path_to_current_experiment, "submission.pkl"), 'wb') as f:
        pickle.dump(predictions_list, f)

print("Run Finished!")