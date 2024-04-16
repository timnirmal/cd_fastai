from fastai.vision.all import *
from fastai.vision.all import PILImage
import os
import torch.nn as nn
import torchvision.models as models
import timm

from utils.data_loader import get_data_loaders_fastai


def load_data(DATASET_PATH):
    train_directory = os.path.join(DATASET_PATH, 'train_dataset')
    valid_directory = os.path.join(DATASET_PATH, 'test_dataset')

    dls = get_data_loaders_fastai(train_directory, valid_directory)
    return dls


def create_densenet_model(model_type="201", num_classes=3):
    if model_type == "201":
        model = models.densenet201(weights=DenseNet201_Weights.DEFAULT)
    elif model_type == "169":
        model = models.densenet169(weights=DenseNet169_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def create_vgg_model(model_type="16", num_classes=3):
    # Load a pre-trained VGG model
    if model_type == "16":
        model = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
    elif model_type == "19":
        model = models.vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    return model


def create_resnet_model(model_type="50", num_classes=3):
    if model_type == "50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    elif model_type == "101":
        model = models.resnet101(weights=ResNet101_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # Replace the fc layer for transfer learning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def create_vit_model(num_classes=3):
    # Load a Vision Transformer model pre-trained on ImageNet
    # Example: 'vit_base_patch16_224' is one of the available Vision Transformer models
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    # Freeze the pretrained weights
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head with a new one suited for your number of classes
    # Note: The number of in_features for the classifier might need to be adjusted based on the ViT model variant
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def create_model_from_name(model_name):
    if model_name == "densenet169_model_01_brain_model":
        return create_densenet_model(model_type="169", num_classes=3)
    elif model_name == "densenet169_model_02_brain_model":
        return create_densenet_model(model_type="169", num_classes=3)
    elif model_name == "densenet201_model_01_brain_model":
        return create_densenet_model(model_type="201", num_classes=3)
    elif model_name == "densenet201_model_02_brain_model":
        return create_densenet_model(model_type="201", num_classes=3)
    elif model_name == "resnet101_model_01_brain_model":
        return create_resnet_model(model_type="101", num_classes=3)
    elif model_name == "resnet101_model_02_brain_model":
        return create_resnet_model(model_type="101", num_classes=3)
    elif model_name == "resnet50_model_01_brain_model":
        return create_resnet_model(model_type="50", num_classes=3)
    elif model_name == "resnet50_model_02_brain_model":
        return create_resnet_model(model_type="50", num_classes=3)
    elif model_name == "vgg16_model_01_brain_model":
        return create_vgg_model(model_type="16", num_classes=3)
    elif model_name == "vgg16_model_02_brain_model":
        return create_vgg_model(model_type="16", num_classes=3)
    elif model_name == "vgg19_model_01_brain_model":
        return create_vgg_model(model_type="19", num_classes=3)
    elif model_name == "vgg19_model_02_brain_model":
        return create_vgg_model(model_type="19", num_classes=3)
    elif model_name == "vision_transformer_model_01_brain_model":
        return create_vit_model(num_classes=3)
    elif model_name == "vision_transformer_model_02_brain_model":
        return create_vit_model(num_classes=3)
    else:
        raise ValueError("Unsupported model name")


def setup_learner(model_name, DATASET_PATH, MODEL_PATH=None):
    dls = load_data(DATASET_PATH)
    model = create_model_from_name(model_name)
    if MODEL_PATH is not None:
        learn = Learner(dls, model, metrics=accuracy, path=MODEL_PATH)
    else:
        learn = Learner(dls, model, metrics=accuracy)
    learn.load(model_name)
    return learn
