import os
import cv2
import pydicom


def convert_dataset_to_jpg(path, save_path):
    # Create the save path directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dcm_files = [f for f in os.listdir(path) if f.endswith('.dcm')]
    for file in dcm_files:
        ds = pydicom.dcmread(os.path.join(path, file))
        pixel_array_numpy = ds.pixel_array

        cv2.imwrite(os.path.join(save_path, file.replace(".dcm", ".jpg")), pixel_array_numpy)


# Paths for the source and destination folders
paths_and_save_paths = [
    ("dataset_01_dcm/Hemorrhagic", "dataset_01_saliency_cam/Hemorrhagic"),
    ("dataset_01_dcm/Ischemic", "dataset_01_saliency_cam/Ischemic"),
    ("dataset_01_dcm/Normal", "dataset_01_saliency_cam/Normal"),
    ("dataset_02_dcm/Acute", "dataset_02_saliency_cam/Acute"),
    ("dataset_02_dcm/Chronic", "dataset_02_saliency_cam/Chronic"),
    ("dataset_02_dcm/Subacute", "dataset_02_saliency_cam/Subacute")
]

# Convert and save the DICOM files as JPG for each specified folder
for path, save_path in paths_and_save_paths:
    convert_dataset_to_jpg(path, save_path)

print("Conversion to JPG completed for all specified folders.")
