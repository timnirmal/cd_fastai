from fastai.vision.all import *
from fastai.vision.all import PILImage
from statistics import mean

from tqdm import tqdm
from tqdm.auto import tqdm

from predict_models.predict_models import setup_learner
from utils.captum_map import generate_layer_conductance
from utils.grad_cam import generate_grad_cam
from utils.saliency_map import generate_saliency_map


def predict_with_models(image_path, models):
    predictions = []
    for model in models:
        img = PILImage.create(image_path)
        _, _, probs = model.predict(img)
        predictions.append(probs.numpy())
    return np.array(predictions)


def initialize_models(dataset, DATASET_01_MODELS, DATASET_02_MODELS, DATASET_01_PATH, DATASET_02_PATH, MODEL_PATH):
    model_names = DATASET_01_MODELS if dataset == 1 else DATASET_02_MODELS
    dataset_path = DATASET_01_PATH if dataset == 1 else DATASET_02_PATH
    models = []
    for model_name in model_names:
        learn = setup_learner(model_name, dataset_path, MODEL_PATH)
        models.append(learn)
    return models


def ensemble_predict(predictions, learner):
    avg_predictions = np.mean(predictions, axis=0)
    final_pred_idx = np.argmax(avg_predictions)
    final_pred = learner.dls.vocab[final_pred_idx]
    final_prob = avg_predictions[final_pred_idx]
    return final_pred, final_prob


DATASET_01_MODELS = [
    'densenet169_model_01_brain_model',
    'densenet201_model_01_brain_model',
    'resnet101_model_01_brain_model',
    'resnet50_model_01_brain_model',
    'vgg16_model_01_brain_model',
    'vgg19_model_01_brain_model',
    'vision_transformer_model_01_brain_model'
]
DATASET_02_MODELS = [
    'densenet169_model_02_brain_model',
    'densenet201_model_02_brain_model',
    'resnet101_model_02_brain_model',
    'resnet50_model_02_brain_model',
    'vgg16_model_02_brain_model',
    'vgg19_model_02_brain_model',
    'vision_transformer_model_02_brain_model'
]


def get_layers(model):
    layers = []
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):  # Example: Filtering only Conv2d layers
            layers.append((name, layer))
    return layers


def process_dataset(dataset_path, dataset,
                    dataset_01_models=DATASET_01_MODELS,
                    dataset_02_models=DATASET_02_MODELS,
                    DATASET_01_PATH=r"datasets\dataset_01",
                    DATASET_02_PATH=r"datasets\dataset_02",
                    MODEL_PATH="",
                    saliency_map=False,
                    grad_cam_map=False
                    ):
    # Ensure model names are in list format
    if isinstance(dataset_01_models, str):
        dataset_01_models = [dataset_01_models]
    if isinstance(dataset_02_models, str):
        dataset_02_models = [dataset_02_models]

    results = []
    image_paths = list(Path(dataset_path).rglob('*.jpg'))

    # Pre-initialize models
    models = initialize_models(dataset, dataset_01_models, dataset_02_models, DATASET_01_PATH, DATASET_02_PATH,
                               MODEL_PATH)

    num_classes = len(models[-1].dls.vocab)

    # keep only one image
    # image_paths = image_paths[3:4]
    for image_path in tqdm(image_paths, desc="Processing images"):
        predictions = predict_with_models(image_path, models)
        if saliency_map:
            for i in range(len(models)):
                model = models[i]
                if dataset == 1:
                    model_name = dataset_01_models[i]
                else:
                    model_name = dataset_02_models[i]
                print(model_name)
                print(image_path.stem)
                save_path = "saliency_maps/"
                save_name = f"{image_path.stem}_{model_name}_saliency_map.jpg"
                print(save_name)
                # generate_saliency_map(model, image_path, save=True, save_path=save_path + save_name)

                all_layers = get_layers(model)
                for layer_name, layer in tqdm(all_layers, desc="Processing Layers"):
                    print(layer)
                    print("\n\n")
                    # # generate_layer_conductance(model, image_path, save=True, save_path=save_path + save_name)
                    # generate_layer_conductance(model, image_path, layer, save=True, save_path=save_path + save_name)
                    # exit()
                    save_path = "grad_cam/"
                    save_name = f"{image_path.stem}_{model_name}_{layer_name}_grad_cam.jpg"
                    generate_grad_cam(model, image_path, layer, save_path=save_path + save_name, save=True, show=False)

        # Use the last model's vocab for ensemble prediction interpretation
        final_pred, final_prob = ensemble_predict(predictions, models[-1])
        row = [image_path.name, *predictions.flatten(), final_pred, final_prob]
        results.append(row)

    print(results)

    return results