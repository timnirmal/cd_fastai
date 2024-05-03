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

# process_dataset("XAI/dataset_01_grad_cam/Hemorrhagic", 1, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)
# # process_dataset("XAI/dataset_01_grad_cam/Ischemic", 1, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)
# process_dataset("XAI/dataset_01_grad_cam/Normal", 1, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)
# process_dataset("XAI/dataset_02_grad_cam/Acute", 2, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)
# process_dataset("XAI/dataset_02_grad_cam/Chronic", 2, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)
# ensemble_result = process_dataset("XAI/dataset_02_grad_cam/Subacute", 2, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)

ensemble_result = process_dataset(r"C:\D\Projects\CD\CD-FastAI\XAI\Required part for the Paper\Model 01", 1, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)
ensemble_result = process_dataset(r"C:\D\Projects\CD\CD-FastAI\XAI\Required part for the Paper\Model 02", 2, saliency_map=True, grad_cam_map=True, dataset_01_models=DATASET_01_MODELS, dataset_02_models=DATASET_02_MODELS)

# process_dataset("XAI/dataset_02_grad_cam/Acute", 2, DATASET_02_MODELS="densenet201_model_02_brain_model")

print("Ensemble Result:")
# print(ensemble_result)