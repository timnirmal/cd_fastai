import os
import random
import shutil

import cv2
import pydicom as dicom
import torch
from fastai.vision.all import *

num_epochs = 3

# GPU/ CPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def convert_dicom_to_jpeg(source_directory, target_directory, image_format='.jpg'):
    """
    Converts DICOM files in the source directory to JPEG format and saves them in the target directory.

    :param source_directory: Directory containing DICOM files.
    :param target_directory: Directory where JPEG files will be saved.
    :param image_format: Format for the converted images (default is .jpg).
    """
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for file_count, filename in enumerate(os.listdir(source_directory)):
        dicom_path = os.path.join(source_directory, filename)
        ds = dicom.dcmread(dicom_path)
        pixel_array_numpy = ds.pixel_array

        image_path = os.path.join(target_directory, f'{file_count}{image_format}')
        cv2.imwrite(image_path, pixel_array_numpy)


def create_test_dataset(source_directory, destination_directory, number_of_files):
    """
    Moves a specified number of random files from the source directory to the destination directory.

    :param source_directory: Directory from which files are to be moved.
    :param destination_directory: Directory to which files are to be moved.
    :param number_of_files: Number of random files to move.
    """
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    files = os.listdir(source_directory)
    selected_files = random.sample(files, min(number_of_files, len(files)))

    for file_name in selected_files:
        shutil.move(os.path.join(source_directory, file_name), destination_directory)


def get_folders(dataset_path):
    """
    Returns a list of folder names in the given dataset path.
    """
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]


def create_folders(folder_path):
    """
    Creates a folder if it doesn't exist.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


# Main script execution
dataset_path = "Model 02 CD"
folders = get_folders(dataset_path)

# create folders for fastai dataset, train dataset, test dataset, and validation dataset
create_folders('fastai_dataset')
create_folders('train_dataset')
create_folders('test_dataset')
create_folders('valid_dataset')

for folder in folders:
    convert_dicom_to_jpeg(os.path.join(dataset_path, folder), os.path.join('fastai_dataset', folder))
    convert_dicom_to_jpeg(os.path.join(dataset_path, folder), os.path.join('train_dataset', folder))

test_size = 200
validation_size = 200

for folder in folders:
    create_test_dataset(os.path.join('train_dataset', folder), os.path.join('test_dataset', folder), test_size)
    create_test_dataset(os.path.join('train_dataset', folder), os.path.join('valid_dataset', folder), validation_size)
