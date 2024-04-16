import os
from pathlib import Path
import cv2
import numpy as np
from fastai.vision.core import PILImage
from matplotlib import pyplot as plt

from predict_models.predict_models import setup_learner

DATASET_01_PATH = r"datasets\model_01_brain"
DATASET_02_PATH = r"datasets\model_02_brain"

# Define a list of model names
model_names_d_01 = [
    'densenet169_model_01_brain_model',
    'densenet201_model_01_brain_model',
    'resnet101_model_01_brain_model',
    'resnet50_model_01_brain_model',
    'vgg16_model_01_brain_model',
    'vgg19_model_01_brain_model',
    'vision_transformer_model_01_brain_model',
]

model_names_d_02 = [
    'densenet169_model_02_brain_model',
    'densenet201_model_02_brain_model',
    'resnet101_model_02_brain_model',
    'resnet50_model_02_brain_model',
    'vgg16_model_02_brain_model',
    'vgg19_model_02_brain_model',
    'vision_transformer_model_02_brain_model'
]

def initialize_models(dataset):
    model_names = model_names_d_01 if dataset == 1 else model_names_d_02
    dataset_path = DATASET_01_PATH if dataset == 1 else DATASET_02_PATH
    models = []
    for model_name in model_names:
        learn = setup_learner(model_name, dataset_path)
        models.append(learn)
    return models


def save_saliency_map(learn, image_path, save_path):
    """
    Generate and save the saliency map for a given image and model.
    """
    learn.model = learn.model.cuda()
    # Similar to the existing generate_saliency_map function but ends with saving the file
    # Ensure the model is in evaluation mode
    learn.model.eval()

    img = PILImage.create(image_path)
    x = learn.dls.test_dl([img]).one_batch()[0]
    x.requires_grad = True

    preds = learn.model(x)
    pred_idx = preds.argmax(dim=1)

    learn.model.zero_grad()
    preds[0, pred_idx].backward()

    gradients = x.grad.data.abs().squeeze().cpu().numpy()
    gradients = np.maximum(gradients, 0)
    grad_max = np.max(gradients)
    if grad_max > 0:
        gradients /= grad_max

    heatmap = np.sum(gradients, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # Create a figure and axis to display the image and color bar
    fig, ax = plt.subplots()
    ax.imshow(superimposed_img)
    ax.axis('off')  # Hide axis

    # Create a color map
    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=255))
    sm.set_array([])  # You have to set a dummy array for this to work

    # Add the color bar
    plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_datasets_for_saliency_maps(base_path):
    """
    Go through each sub-folder in the base_path, detect the dataset type based on the folder structure,
    and apply the saliency function with each model on each image, saving the outputs.
    """
    # Mapping from sub-folder names to dataset identifiers
    dataset_mapping = {
        "Hemorrhagic": 1, "Ischemic": 1, "Normal": 1,
        "Acute": 2, "Chronic": 2, "Subacute": 2
    }

    # Iterate over the sub-folders in the base path
    for sub_folder in Path(base_path).rglob("*"):
        if sub_folder.is_dir() and sub_folder.parent.name.startswith("Dataset 02 - color bar"):
            # Determine the dataset based on the sub-folder name
            dataset = dataset_mapping.get(sub_folder.name)
            if dataset is None:
                continue  # Skip if the sub-folder is not part of the mapping

            # Select the correct model names and dataset path based on the dataset identifier
            model_names = model_names_d_01 if dataset == 1 else model_names_d_02
            dataset_path = DATASET_01_PATH if dataset == 1 else DATASET_02_PATH

            print(f"Processing {sub_folder}")

            # Initialize models for the current dataset
            models = initialize_models(str(dataset))

            # Process each image in the sub-folder
            for image_path in sub_folder.glob('*.jpg'):
                for model, model_name in zip(models, model_names):
                    saliency_save_path = str(image_path).replace('.jpg', f'_{model_name}_saliency.jpg')
                    save_saliency_map(model, str(image_path), saliency_save_path)
                    print(f"Saliency map saved to {saliency_save_path}")


# Example base path that contains the Dataset folders
BASE_PATH = "For XAI Saliency and Grad-CAM"

if __name__ == '__main__':
    process_datasets_for_saliency_maps(BASE_PATH)
