import argparse
import json
from predict_models.fastai_models import process_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some datasets.")
    parser.add_argument('-dir', '--directory', required=True, help='Directory for the model')
    parser.add_argument('-set', '--dataset', type=int, required=True, help='Dataset index')
    parser.add_argument('-sal', '--saliency', action='store_true', help='Use saliency map')
    parser.add_argument('-gc', '--gradcam', action='store_true', help='Use Grad-CAM')
    parser.add_argument('-m1', '--models1', type=str, required=True, help='JSON list of models for dataset 1')
    parser.add_argument('-m2', '--models2', type=str, required=True, help='JSON list of models for dataset 2')

    args = parser.parse_args()

    DATASET_01_MODELS = json.loads(args.models1)
    DATASET_02_MODELS = json.loads(args.models2)

    process_dataset(
        args.directory,
        args.dataset,
        saliency_map=args.saliency,
        grad_cam_map=args.gradcam,
        dataset_01_models=DATASET_01_MODELS,
        dataset_02_models=DATASET_02_MODELS
    )
