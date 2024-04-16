import os
from pathlib import Path
import cv2
import numpy as np
import torch
from fastai.vision.core import PILImage
from matplotlib import pyplot as plt

from ensemble_averaging_for_test_dataset_fast import initialize_models

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


def save_grad_cam(learn, image_path, save_path):
    learn.model.eval().cuda()

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)
        return None

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        return None

    # Use register_full_backward_hook for PyTorch 1.8 and later
    target_layer = list(learn.model.modules())[-1]
    target_layer.register_forward_hook(forward_hook)
    hook = target_layer.register_full_backward_hook(backward_hook)

    # Process the image
    img_tensor = PILImage.create(image_path)
    img_tensor = learn.dls.test_dl([img_tensor]).one_batch()[0].cuda()

    # Forward pass
    output = learn.model(img_tensor)
    one_hot_output = torch.zeros((1, output.size()[-1]), device="cuda")
    one_hot_output[0][output.argmax(1)] = 1

    # Backward pass
    learn.model.zero_grad()
    output.backward(gradient=one_hot_output, retain_graph=True)

    # Remove the hook
    hook.remove()

    # Check that gradients have been captured
    if len(gradients) == 0 or gradients[0] is None:
        raise RuntimeError("No gradients captured - check backward hook registration.")

    # Gradients and activations
    gradient = gradients[0].detach().cpu().numpy().squeeze()
    activation = activations[0].detach().cpu().numpy().squeeze()

    # Weighting the channels by the gradients
    if gradient.ndim == 3 and activation.ndim == 3:
        weights = np.mean(gradient, axis=(1, 2))
        grad_cam = np.dot(activation.transpose(1, 2, 0), weights).transpose(2, 0, 1)
    elif gradient.ndim == 1 and activation.ndim == 2:
        # This assumes a linear layer follows the convolution layers
        grad_cam = gradient[:, None, None] * activation
    else:
        raise ValueError(f"Unexpected shapes - grad: {gradient.shape}, act: {activation.shape}")

    # Applying ReLU to the Grad-CAM
    grad_cam = np.maximum(grad_cam, 0)
    grad_cam = cv2.resize(grad_cam, (img_tensor.shape[-1], img_tensor.shape[-2]))
    grad_cam = grad_cam - np.min(grad_cam)
    grad_cam /= np.max(grad_cam)

    # Reading the original image
    original_img = cv2.imread(str(image_path))
    heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    superimposed_img = heatmap * 0.5 + original_img

    # Saving the final image
    cv2.imwrite(save_path, superimposed_img)



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
        if sub_folder.is_dir() and sub_folder.parent.name.startswith("Dataset 01 Grad Cam"):
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
                    save_grad_cam(model, image_path, saliency_save_path)
                    print(f"Saliency map saved to {saliency_save_path}")


# Example base path that contains the Dataset folders
BASE_PATH = "For XAI Saliency and Grad-CAM"

if __name__ == '__main__':
    process_datasets_for_saliency_maps(BASE_PATH)
