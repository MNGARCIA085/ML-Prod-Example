import argparse
from datetime import datetime
from src.config.constants import BREAST_CANCER_CSV_RAW
from src.tuning.workflow import get_best_model
from src.experiments.models import save_best_model
from src.tuning.utils import save_logs
from src.data.utils import get_preprocessors
from src.models.factory import get_model_fns_tuner


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Tune models with different data preprocessing variants.")

    # Optional selection of models and data variants
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["baseline", "no_dropout", "dropout"],
        help="Models to tune (default: all)"
    )
    parser.add_argument(
        "--data_variants",
        nargs="+",
        choices=["simple", "standardize"],
        help="Data preprocessing variants to use (default: all)"
    )

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for preprocessing (default: 64)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for tuning (default: 2)")
    parser.add_argument("--max_trials", type=int, default=1, help="Max tuning trials (default: 1)")

    args = parser.parse_args()

    # Filepath
    filepath = BREAST_CANCER_CSV_RAW

    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # preprocessors
    all_variants = ["simple", "standardize"]
    selected_variants = args.data_variants or all_variants
    preprocessors = get_preprocessors(selected_variants, args.batch_size)


    # Models
    all_models = ["baseline", "no_dropout", "dropout"]
    selected_models = args.models or all_models
    model_fns = get_model_fns_tuner()
    # filter only selected models
    model_fns = {name: fn for name, fn in model_fns.items() if name in selected_models}
    
    # get best (model + prep)
    results, best_model_info, preprocessor, recall, best_f1 = get_best_model(preprocessors, model_fns, filepath, max_trials=1, 
                                                               epochs=args.epochs, recall_threshold=0.2)


    # Save the best model automatically
    save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor)

    # Save all results into a single JSON
    save_logs(results, timestamp)

    # return
    return best_model_info
    


if __name__ == "__main__":
    best_model_info = main()




# Note. I might not have a best model if none of them is better than the recall threshold
# python -m scripts.tuning --models baseline dropout --data_variants simple
