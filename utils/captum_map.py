from captum.attr import LayerConductance
import torch
from fastai.vision.core import PILImage
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path

def generate_layer_conductance(learn, image_path, target_layer, save_path="layer_conductance", save=False, show=False):
    # Ensure the model is in evaluation mode
    learn.model.eval()
    device = next(learn.model.parameters()).device  # Get the device of the model

    # Convert the image to a tensor
    img = PILImage.create(image_path)
    x = learn.dls.test_dl([img]).one_batch()[0].to(device)
    x = x.detach().clone()
    x.requires_grad = True

    target_layer = learn.model[0][0] if isinstance(learn.model, torch.nn.Sequential) else learn.model.conv1
    
    # Forward pass
    preds = learn.model(x)
    target_class = preds.argmax(dim=1).item()

    # Initialize Layer Conductance
    lc = LayerConductance(learn.model, target_layer)

    # Compute attributions using Layer Conductance
    attributions_lc = lc.attribute(x, target=target_class)
    attributions_lc = attributions_lc.mean(dim=1).squeeze().cpu().detach().numpy()  # Take mean across color channels

    # Normalize attributions
    attributions_lc = np.maximum(attributions_lc, 0)
    attributions_max = np.max(attributions_lc)
    if attributions_max > 0:
        attributions_lc /= attributions_max

    # Convert WindowsPath to string if necessary and read the original image for overlay
    image_path_str = str(image_path)
    original_img = cv2.imread(image_path_str)
    if original_img is None:
        raise ValueError(f"Failed to load image from path: {image_path_str}")

    # Resize heatmap to match the original image size
    heatmap = cv2.resize(attributions_lc, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on original image
    superimposed_img = heatmap * 0.4 + original_img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    # Convert BGR to RGB for matplotlib visualization
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(original_img_rgb)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    ax[1].imshow(superimposed_img_rgb / superimposed_img_rgb.max())
    ax[1].axis('off')
    ax[1].set_title('Conductance Map Overlay')

    ax[2].imshow(heatmap, cmap='hot')
    ax[2].axis('off')
    ax[2].set_title('Conductance Map')

    if save:
        # create the save directory if it does not exist
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + ".png")

    if show:
        plt.show()

