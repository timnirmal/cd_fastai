from pathlib import Path

from fastai.vision.core import PILImage
import matplotlib.pyplot as plt
import numpy as np
import cv2


def generate_saliency_map(learn, image_path, save_path="saliency_map", save=False, show=False):
    # Ensure the model is in evaluation mode
    learn.model.eval()

    # Convert the image to a tensor and add a batch dimension
    img = PILImage.create(image_path)
    x = learn.dls.test_dl([img]).one_batch()[0]
    x.requires_grad = True

    # Forward pass
    preds = learn.model(x)
    pred_idx = preds.argmax(dim=1)

    # Backward pass for the correct class
    learn.model.zero_grad()
    preds[0, pred_idx].backward()

    # Extract gradients
    gradients = x.grad.data.abs().squeeze().cpu().numpy()
    gradients = np.maximum(gradients, 0)
    grad_max = np.max(gradients)
    if grad_max > 0:
        gradients /= grad_max

    # Convert gradients to heatmap
    heatmap = np.sum(gradients, axis=0)  # Summing across color channels
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # Normalize

    # Convert WindowsPath to string if necessary
    image_path_str = str(image_path)
    # print(f"Debug: image_path type: {type(image_path_str)}, value: {image_path_str}")

    # Read the original image for overlay
    img = cv2.imread(image_path_str)
    if img is None:
        raise ValueError(f"Failed to load image from path: {image_path_str}")

    # Read the original image for overlay
    # img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    # Convert BGR to RGB for matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    superimposed_img_2 = np.clip(superimposed_img, 0, 255).astype('uint8')

    # Now, it's safe to convert color spaces
    superimposed_img_2 = cv2.cvtColor(superimposed_img_2, cv2.COLOR_BGR2RGB)

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    ax[1].imshow(superimposed_img_2 / superimposed_img_2.max())
    ax[1].axis('off')
    ax[1].set_title('Saliency Map Overlay')
    # fig.colorbar(ax[1].imshow(superimposed_img_2 / superimposed_img_2.max()), ax=ax[1])

    ax[2].imshow(heatmap, cmap='hot')
    ax[2].axis('off')
    ax[2].set_title('Saliency Map')
    # fig.colorbar(ax[2].imshow(heatmap, cmap='hot'), ax=ax[2])

    if save:
        # create the save directory if it does not exist
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path + ".png")

    if show:
        plt.show()
