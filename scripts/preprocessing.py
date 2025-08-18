import argparse
import tensorflow as tf
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.config.constants import BREAST_CANCER_CSV_RAW


def print_first_element(ds, name: str):
    """Prints the first element of the first batch (features and label)."""
    for x_batch, y_batch in ds.take(1):
        print(f"{name} - first element of first batch:")
        print("Features:", x_batch.numpy()[0])
        print("Label:", y_batch.numpy()[0])
    print("-" * 40)


def main(prep_style: str = None):
    styles_to_run = []

    if prep_style is None:
        # Run both if no flag
        styles_to_run = ["not_standardize", "standardize"]
    else:
        if prep_style not in ["standardize", "not_standardize"]:
            raise ValueError("prep_style must be 'standardize' or 'not_standardize'")
        styles_to_run = [prep_style]

    for style in styles_to_run:
        if style == "standardize":
            preprocessor = BreastCancerPreprocessorNormalized(batch_size=16)
            print("\n=== Using standardized preprocessing ===")
        else:
            preprocessor = BreastCancerPreprocessor(batch_size=16)
            print("\n=== Using non-standardized preprocessing ===")

        train_ds, val_ds, test_ds = preprocessor.get_datasets(BREAST_CANCER_CSV_RAW)

        # Print only the first element of the first batch
        print_first_element(train_ds, "Train")
        print_first_element(val_ds, "Validation")
        print_first_element(test_ds, "Test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Breast Cancer TF datasets with optional preprocessing.")
    parser.add_argument(
        "--prep_style",
        type=str,
        choices=["standardize", "not_standardize"],
        default=None,
        help="Choose preprocessing style: 'standardize', 'not_standardize', or leave empty for both."
    )
    args = parser.parse_args()
    main(args.prep_style)



#python3.10 -m scripts.preprocessing
#python3.10 -m scripts.preprocessing --prep_style standardize
