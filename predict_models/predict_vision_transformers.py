# import os
#
# import timm
# from Model_Training.load_data import get_data_loaders_fastai
# from fastai.vision.all import *
# from fastai.vision.all import PILImage
#
# model_name = 'vision_transformer_model_01_brain_model'
# DATASET_PATH = r"..\datasets\model_02_brain"
#
#
# def create_vit_model(num_classes=3):
#     # Load a Vision Transformer model pre-trained on ImageNet
#     # Example: 'vit_base_patch16_224' is one of the available Vision Transformer models
#     model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
#
#     # Freeze the pretrained weights
#     for param in model.parameters():
#         param.requires_grad = False
#
#     # Replace the classifier head with a new one suited for your number of classes
#     # Note: The number of in_features for the classifier might need to be adjusted based on the ViT model variant
#     model.head = nn.Linear(model.head.in_features, num_classes)
#     return model
#
#
# def load_data():
#     train_directory = os.path.join(DATASET_PATH, 'train_dataset')
#     valid_directory = os.path.join(DATASET_PATH, 'test_dataset')
#
#     dls = get_data_loaders_fastai(train_directory, valid_directory)
#     return dls
#
#
# dls = load_data()
# model = create_vit_model(num_classes=3)
# learn = Learner(dls, model, metrics=accuracy)
# learn.load(model_name)
#
# # Path to the image you want to predict
# # image_path = '../datasets/model_02_brain/test_dataset/Acute/2.jpg'
# image_path = '../datasets/model_02_brain/test_dataset/Chronic/223.jpg'
#
# # Load the image
# img = PILImage.create(image_path)
#
# # Perform inference
# pred, pred_idx, probs = learn.predict(img)
#
# # Print results
# print(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
#
#
# def generate_saliency_map(learn, image_path):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import cv2
#
#     # Ensure the model is in evaluation mode
#     learn.model.eval()
#
#     # Convert the image to a tensor and add a batch dimension
#     img = PILImage.create(image_path)
#     x = learn.dls.test_dl([img]).one_batch()[0]
#     x.requires_grad = True
#
#     # Forward pass
#     preds = learn.model(x)
#     pred_idx = preds.argmax(dim=1)
#
#     # Backward pass for the correct class
#     learn.model.zero_grad()
#     preds[0, pred_idx].backward()
#
#     # Extract gradients
#     gradients = x.grad.data.abs().squeeze().cpu().numpy()
#     gradients = np.maximum(gradients, 0)
#     grad_max = np.max(gradients)
#     if grad_max > 0:
#         gradients /= grad_max
#
#     # Convert gradients to heatmap
#     heatmap = np.sum(gradients, axis=0)  # Summing across color channels
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)  # Normalize
#
#     # Read the original image for overlay
#     img = cv2.imread(image_path)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#
#     superimposed_img = heatmap * 0.4 + img
#
#     # Convert BGR to RGB for matplotlib
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     superimposed_img_2 = np.clip(superimposed_img, 0, 255).astype('uint8')
#
#     # Now, it's safe to convert color spaces
#     superimposed_img_2 = cv2.cvtColor(superimposed_img_2, cv2.COLOR_BGR2RGB)
#
#     # Plotting
#     fig, ax = plt.subplots(1, 3, figsize=(12, 6))
#     ax[0].imshow(img)
#     ax[0].axis('off')
#     ax[0].set_title('Original Image')
#
#     ax[1].imshow(superimposed_img_2 / superimposed_img_2.max())
#     ax[1].axis('off')
#     ax[1].set_title('Saliency Map Overlay')
#     # fig.colorbar(ax[1].imshow(superimposed_img_2 / superimposed_img_2.max()), ax=ax[1])
#
#     ax[2].imshow(heatmap, cmap='hot')
#     ax[2].axis('off')
#     ax[2].set_title('Saliency Map')
#     # fig.colorbar(ax[2].imshow(heatmap, cmap='hot'), ax=ax[2])
#
#     plt.show()
#
#
# generate_saliency_map(learn, image_path)
