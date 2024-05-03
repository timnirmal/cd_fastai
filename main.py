import argparse

from predict_models.fastai_models import process_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some datasets.")
    parser.add_argument('-dir', '--directory', required=True, help='Directory to process')
    parser.add_argument('-m', '--model', required=True, help='Model to use')
    parser.add_argument('-sal', '--saliency', action='store_true', help='Use saliency map')
    parser.add_argument('-gc', '--gradcam', action='store_true', help='Use Grad-CAM')

    args = parser.parse_args()

    ensemble_result = process_dataset(
        args.directory,
        1,  # Or any other logic you need
        saliency_map=args.saliency,
        grad_cam_map=args.gradcam,
        dataset_01_models=args.model,
        dataset_02_models=args.model
    )

    print("Ensemble Result:")
    print(ensemble_result)
