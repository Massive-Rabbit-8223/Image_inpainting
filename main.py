from pickle import TRUE
import os
import splitfolders

from utils.validate import validate_images
from dataset_generator import create_dataset


config = {
    "Data": {
        "raw_img_dir": os.path.abspath(r"../../training"),
        "data_dir": os.path.abspath(r"../../Data/"),
        "valid_img_folder": r"Valid_data",
        "log_file": "logfile.log",
        "formatter": "06d",
        "train_val_split_folder": r"Dataset",
        "seed": 1337,
        "train_val_split_ratio": (0.8, 0.2)
    },
    "Model": {

    },
    "Training": {

    }
}

FILTER_IMAGES = True
CREATE_DATASET = True
TRAIN_MODEL = False
LOG_MODEL = False

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
    
    train_input_dir = os.path.join(os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), "train/")
    val_input_dir = os.path.join(os.path.join(config["Data"]["data_dir"], config["Data"]["train_val_split_folder"]), "val/")

    print("train_input_dir: ", train_input_dir)
    print("val_input_dir: ", val_input_dir)

    train_dataset, val_dataset = create_dataset(
        (train_input_dir, val_input_dir)
    )

    print(f"train_dataset length: {len(train_dataset)}")
    print(f"val_dataset length: {len(val_dataset)}")

model = None

if TRAIN_MODEL == True:
    pass


