import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from fastai.vision.all import *

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from fastai.vision.all import *

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def generate_grad_cam(learn, image_path, layer, save_path="grad_cam", save=False, show=False):
    # Ensure model is in evaluation mode
    # learn.model.eval()

    torch.cuda.empty_cache()  # Clear memory cache
    learn.model.eval()
    learn.dls.device = 'cuda'  # Ensure we're using GPU

    # Hook to capture the outputs of the target layer
    def forward_hook(module, input, output):
        global feature_maps
        feature_maps = output.detach()

    # Hook to capture the gradients
    def backward_hook(module, grad_in, grad_out):
        global gradients
        gradients = grad_out[0].detach()

    # Register hooks
    handle_f = layer.register_forward_hook(forward_hook)
    handle_b = layer.register_full_backward_hook(backward_hook)

    # Convert the image to a tensor
    img = PILImage.create(image_path)
    x = learn.dls.test_dl([img]).one_batch()[0]
    x.requires_grad = True

    # Forward pass to get model outputs
    preds = learn.model(x)
    pred_idx = preds.argmax(dim=1)

    # Backward pass for the correct class
    learn.model.zero_grad()
    preds[:, pred_idx].backward()

    # Calculate mean of gradients for each filter
    pooled_gradients = torch.mean(gradients, [0, 2, 3])

    # Weight feature maps
    weighted_feature_maps = feature_maps[0] * pooled_gradients[:, None, None]

    # Create the Grad-CAM heatmap
    grad_cam_map = weighted_feature_maps.sum(dim=0).clamp(min=0).cpu()
    grad_cam_map = F.interpolate(grad_cam_map.unsqueeze(0).unsqueeze(0), x.shape[-2:], mode='bilinear',
                                 align_corners=False).squeeze().numpy()
    # Cleanup hooks
    handle_f.remove()
    handle_b.remove()

    # Read the original image
    image = cv2.imread(str(image_path))
    heatmap = cv2.resize(grad_cam_map, (image.shape[1], image.shape[0]))
    heatmap = np.maximum(heatmap, 0)  # Remove negative values
    heatmap = heatmap / np.max(heatmap)  # Normalize to range [0, 1]
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + image
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


    # Plotting
    if show:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(superimposed_img)
        plt.title('Grad-CAM')
        plt.axis('off')
        plt.show()

    if save:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{save_path}_grad_cam.jpg", cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return grad_cam_map

