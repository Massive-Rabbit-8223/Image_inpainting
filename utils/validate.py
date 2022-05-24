"""
Author: Patrick Mederitsch
Matr.Nr.: K51831808
Exercise 2
"""

import string
from PIL import Image
import os
import numpy as np
import hashlib
import shutil


def validity_check(image_file_path: string, hash_list: list):
    """
    conditions:
    1. The file name ends with .jpg, .JPG, .jpeg or .JPEG.
    2. The file size does not exceed 250kB (=250 000 Bytes).
    3. The file can be read as image (i.e., the PIL/pillow module does not raise an exception
    when reading the file).
    4. The image data has a shape of (H, W, 3) with H (height) and W (width) larger than or
    equal to 96 pixels. The three channels must be in the order RGB (red, green, blue).
    5. The image data has a variance larger than 0, i.e., there is not just one common RGB pixel
    in the image data.
    6. The same image has not been copied already.

    return:
    error code if conditions are  violated [1,2,3,4,5,6] or zero otherwise (integer)
    list of image hashes
    """
    error_code = 0

    _, file_ending = os.path.splitext(image_file_path)
    file_name = image_file_path
    file_size = os.path.getsize(file_name)  # size in bytes

    ### 1. check file ending ###
    if not file_ending in [".jpg", ".JPG", ".jpeg", ".JPEG"]:
        error_code = 1
        return error_code, hash_list
    ### 2. check file size ###
    if file_size > 250000:
        error_code = 2
        return error_code, hash_list
    ### 3. check if file can be read as as image ###
    try:
        image = Image.open(image_file_path)
    except:
        error_code = 3
        return error_code, hash_list
    ### 4. check image shape and channel order ##
    image_as_array = np.array(image)
    if (image.mode != "RGB") or (image.height < 96) or (image.width < 96) or (image_as_array.shape[-1] != 3):
        error_code = 4
        return error_code, hash_list
    ### 5. check if variance is large than 0 ###
    if (image_as_array[:,:,0].var() <= 0) and (image_as_array[:,:,1].var() <= 0) and (image_as_array[:,:,2].var() <= 0):
        error_code = 5
        return error_code, hash_list
    ### 6. check if image has been copied already
    with open(image_file_path, "rb") as f:
        hash = hashlib.sha256(f.read()).hexdigest()
    if hash in hash_list:
        error_code = 6
        return error_code, hash_list
    else:
        hash_list.append(hash)
    
    return error_code, hash_list

def validate_images(input_dir: string, output_dir: string, log_file: string, formatter="01d"):
    """
    copy valid files into output_dir (string)

    parameters:
    input_dir (string): The input directory where your function should look for files.
    output_dir (string): The directory where your function should place the output files.
    log_file (string): The path of the log file.
    formatter (string, optional): Optional format string to use when writing the output files.
    It expects a single number that is used to apply the format string on. The result is the
    base name of the file, without any extension.

    return:
    number of valid filed that were copied (integer)
    """
    log_filepath, log_filename = os.path.split(log_file)
    ### check if directories exist, otherwise create them ###
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_filepath):
        os.makedirs(log_filepath)

    ### create absolute paths ###
    absolute_input_dir = os.path.abspath(input_dir)
    absolute_output_dir = os.path.abspath(output_dir)
    absolute_log_dir = os.path.abspath(log_filepath)

    log_list = []
    valid_image_list = []
    hash_list = []

    ### filter images, store valid images in the valid_image_list and invalid image names in the log_list ###
    for root,d_names,f_names in os.walk(absolute_input_dir):
        for f in f_names:
            image_file_path = os.path.join(root, f)
            error_code, hash_list = validity_check(image_file_path, hash_list)

            if error_code == 0:
                valid_image_list.append(image_file_path)
            else:
                _, filename = os.path.split(image_file_path)
                log_message = filename+";"+str(error_code)+"\n"
                log_list.append(log_message)

    ### write valid images to output_dir using given format or default format ###
    image_counter = 0
    for valid_image_path in sorted(valid_image_list):
        output_file = os.path.join(absolute_output_dir, str(format(image_counter, formatter)) + ".jpg")
        try:
            shutil.copy(valid_image_path, output_file)
            print(f"File {image_counter}.jpg copied successfully.")
            image_counter += 1
        # If source and destination are same
        except shutil.SameFileError:
            print("Source and destination represents the same file.")
        # If there is any permission issue
        except PermissionError:
            print("Permission denied.")
        # For other errors
        except:
            print("Error occurred while copying file.")

    ### write log messages into logfile ###
    with open(os.path.join(absolute_log_dir, log_filename), "w") as f:
        for log_message in sorted(log_list):
            f.write(log_message)

    return image_counter
        
if __name__ == "__main__":
    input_dir = os.path.abspath(r"../training")
    output_dir = os.path.abspath(r"../Data/Valid_training")
    logfile = os.path.abspath(r"../Data/logfile.log")
    formatter = "06d"

    print(input_dir)

    print(validate_images(input_dir, output_dir, logfile, formatter))