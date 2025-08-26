import argparse
import json
import os
from datetime import datetime
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.models.model_dropout import build_model_with_dropout_tuner
from src.models.model_no_dropout import build_model_no_dropout_tuner
from src.models.baseline import baseline_with_tuner
from src.config.constants import BREAST_CANCER_CSV_RAW
from src.tuning.workflow import tune_models, get_best_model
from src.experiments.models import save_best_model
from src.tuning.utils import save_logs




# MAIN
def main():
    parser = argparse.ArgumentParser(description="Tune multiple models with flags.")
    parser.add_argument("--baseline", action="store_true", help="Tune baseline model")
    parser.add_argument("--no_dropout", action="store_true", help="Tune model without dropout")
    parser.add_argument("--dropout", action="store_true", help="Tune model with dropout")
    parser.add_argument("--max_trials", type=int, default=3, help="Max trials for Keras Tuner")
    parser.add_argument("--epochs", type=int, default=5, help="Max epochs per trial") #better default: 20
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")
    args = parser.parse_args()

    tune_all = not (args.baseline or args.no_dropout or args.dropout)

    # Prepare data
    filepath = BREAST_CANCER_CSV_RAW
    preprocessor = BreastCancerPreprocessorNormalized(batch_size=args.batch_size)
    train_ds, val_ds = preprocessor.get_datasets(filepath)

    models_to_tune = []
    if args.baseline or tune_all:
        models_to_tune.append(("baseline", baseline_with_tuner))
    if args.no_dropout or tune_all:
        models_to_tune.append(("no_dropout", build_model_no_dropout_tuner))
    if args.dropout or tune_all:
        models_to_tune.append(("dropout", build_model_with_dropout_tuner))


    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # get best model
    results, best_model_info, recall, best_f1 = get_best_model(train_ds, val_ds, args.max_trials,args.epochs, models_to_tune)

    # Save the best model automatically
    save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor)

    # Save all results into a single JSON
    save_logs(results, timestamp)

    # return
    return best_model_info
    


if __name__ == "__main__":
    best_model_info = main()




# Note. I might not have a best model if none of them is better than the recall threshold