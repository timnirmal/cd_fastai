from predict_models.fastai_models import process_dataset

MODEL_PATH = "../models"

DATASET_01_PATH = r"datasets\dataset_01"
DATASET_02_PATH = r"datasets\dataset_02"

# Define a list of model names
DATASET_01_MODELS = [
    # 'densenet169_model_01_brain_model',
    'densenet201_model_01_brain_model',
    # 'resnet101_model_01_brain_model',
    # 'resnet50_model_01_brain_model',
    # 'vgg16_model_01_brain_model',
    # 'vgg19_model_01_brain_model',
    # 'vision_transformer_model_01_brain_model',
]

DATASET_02_MODELS = [
    # 'densenet169_model_02_brain_model',
    'densenet201_model_02_brain_model',
    # 'resnet101_model_02_brain_model',
    # 'resnet50_model_02_brain_model',
    # 'vgg16_model_02_brain_model',
    # 'vgg19_model_02_brain_model',
    # 'vision_transformer_model_02_brain_model'
]

ensemble_result = process_dataset("XAI/dataset_01_grad_cam/Hemorrhagic", 1, saliency_map=True)
# process_dataset("XAI/dataset_02_grad_cam/Acute", 2, DATASET_02_MODELS="densenet201_model_02_brain_model")

print("Ensemble Result:")
print(ensemble_result)