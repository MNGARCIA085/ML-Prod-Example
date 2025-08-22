import argparse
import json
import joblib
import os
from datetime import datetime
from sklearn.metrics import f1_score
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.models.model_dropout import build_model_with_dropout_tuner
from src.models.model_no_dropout import build_model_no_dropout_tuner
from src.models.baseline import baseline_with_tuner
from src.tuning.tuner import ModelTuner
from src.config.constants import BREAST_CANCER_CSV_RAW


# save results
def save_logs(results, timestamp):
    os.makedirs("logs/tuning", exist_ok=True)
    json_file = f"logs/tuning/all_models_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All models results saved to {json_file}")




# save best model
def save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor=None):
    os.makedirs("outputs/saved_models", exist_ok=True)

    # Create experiment folder
    exp_dir = f"outputs/saved_models/experiment_{best_model_info['name']}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(exp_dir, "model.h5")
    best_model_info["model"].save(model_path)
    print(f"Model saved to {model_path}")

    # Save scaler (if exists)
    if preprocessor and getattr(preprocessor, "scaler", None) is not None:
        scaler_path = os.path.join(exp_dir, "scaler.pkl")
        joblib.dump(preprocessor.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    # Save encoder (if exists)
    if preprocessor and getattr(preprocessor, "encoder", None) is not None:
        encoder_path = os.path.join(exp_dir, "encoder.pkl")
        joblib.dump(preprocessor.encoder, encoder_path)
        print(f"Encoder saved to {encoder_path}")

    # Save feature order (columns)
    if preprocessor and getattr(preprocessor, "feature_columns", None) is not None:
        columns_path = os.path.join(exp_dir, "columns.json")
        with open(columns_path, "w") as f:
            json.dump(preprocessor.feature_columns, f, indent=4)
        print(f"Feature columns saved to {columns_path}")

    # Save small metadata file (optional but useful)
    meta_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "name": best_model_info["name"],
            "best_hyperparameters": best_model_info["best_hyperparameters"],
            "val_metrics": best_model_info["val_metrics"],
            "f1_score": best_f1,
            "recall": recall
        }, f, indent=4)
    print(f"Metadata saved to {meta_path}")

    print(f"\nBest model according to our criterion: {best_model_info['name']}, "
          f"F1={best_f1:.4f}, Recall={recall:.4f}")




# get the best model and its data
def get_best_modelv0(train_ds, val_ds, max_trials, epochs, models_to_tune, recall_threshold=0.2):
    results = {}
    best_model_info = None
    best_f1 = -1.0
    best_model_recall = -1.0  # track recall of the best model

    for model_name, build_fn in models_to_tune:
        print(f"\n=== Tuning {model_name} model ===")
        tuner = ModelTuner(build_model_fn=build_fn)
        best_model, best_hp, val_metrics = tuner.run(
            train_ds,
            val_ds,
            max_trials=max_trials,
            epochs=epochs
        )

        results[model_name] = {
            "best_hyperparameters": best_hp.values,
            "val_metrics": val_metrics,
        }

        recall = val_metrics.get("recall", 0.0)
        f1 = val_metrics.get("f1_score", 0.0)

        if recall >= recall_threshold:
            if f1 > best_f1:
                best_f1 = f1
                best_model_recall = recall
                best_model_info = {
                    "name": model_name,
                    "model": best_model,
                    "val_metrics": val_metrics,
                    "best_hyperparameters": best_hp.values
                }

        print(f"{model_name} best hyperparameters: {best_hp.values}")
        print(f"{model_name} val metrics: {val_metrics}\n")

    return results, best_model_info, best_model_recall, best_f1






def tune_models(train_ds, val_ds, max_trials, epochs, models_to_tune):
    """
    Run tuning for each model and return JSON-serializable results 
    (no Keras models included).
    """
    results = {}
    models = {}  # keep actual models separately

    for model_name, build_fn in models_to_tune:
        print(f"\n=== Tuning {model_name} model ===")
        tuner = ModelTuner(build_model_fn=build_fn)
        best_model, best_hp, val_metrics = tuner.run(
            train_ds,
            val_ds,
            max_trials=max_trials,
            epochs=epochs
        )

        # JSON-safe results
        results[model_name] = {
            "best_hyperparameters": best_hp.values,
            "val_metrics": val_metrics,
        }

        # keep model in a separate dict
        models[model_name] = best_model

        print(f"{model_name} best hyperparameters: {best_hp.values}")
        print(f"{model_name} val metrics: {val_metrics}\n")

    return results, models


def select_best_model(results, models, recall_threshold=0.2):
    """
    Choose the best model according to recall + F1.
    Only here we attach the actual model object.
    """
    best_model_info = None
    best_f1 = -1.0
    best_model_recall = -1.0

    for model_name, info in results.items():
        recall = info["val_metrics"].get("recall", 0.0)
        f1 = info["val_metrics"].get("f1_score", 0.0)

        if recall >= recall_threshold and f1 > best_f1:
            best_f1 = f1
            best_model_recall = recall
            best_model_info = {
                "name": model_name,
                "model": models[model_name],  # attach model here
                "val_metrics": info["val_metrics"],
                "best_hyperparameters": info["best_hyperparameters"]
            }

    return best_model_info, best_model_recall, best_f1


def get_best_model(train_ds, val_ds, max_trials, epochs, models_to_tune, recall_threshold=0.2):
    results, models = tune_models(train_ds, val_ds, max_trials, epochs, models_to_tune)
    best_model_info, best_model_recall, best_f1 = select_best_model(results, models, recall_threshold)
    return results, best_model_info, best_model_recall, best_f1









# MAIN
def main():
    parser = argparse.ArgumentParser(description="Tune multiple models with flags.")
    parser.add_argument("--baseline", action="store_true", help="Tune baseline model")
    parser.add_argument("--no_dropout", action="store_true", help="Tune model without dropout")
    parser.add_argument("--dropout", action="store_true", help="Tune model with dropout")
    parser.add_argument("--max_trials", type=int, default=3, help="Max trials for Keras Tuner")
    parser.add_argument("--epochs", type=int, default=5, help="Max epochs per trial") #better default: 20
    args = parser.parse_args()

    tune_all = not (args.baseline or args.no_dropout or args.dropout)

    # Prepare data
    filepath = BREAST_CANCER_CSV_RAW
    preprocessor = BreastCancerPreprocessorNormalized(batch_size=32)
    train_ds, val_ds = preprocessor.get_datasets(filepath)

    models_to_tune = []
    if args.baseline or tune_all:
        models_to_tune.append(("baseline", baseline_with_tuner))
    if args.no_dropout or tune_all:
        models_to_tune.append(("no_dropout", build_model_no_dropout_tuner))
    if args.dropout or tune_all:
        models_to_tune.append(("dropout", build_model_with_dropout_tuner))


    # get best model
    #results, best_model_info, recall, best_f1 = get_best_model(train_ds, val_ds, args.max_trials,args.epochs, models_to_tune)
    results, models = tune_models(train_ds, val_ds, args.max_trials, args.epochs, models_to_tune)
    best_model_info, recall, best_f1 = select_best_model(results, models, recall_threshold=0.2)


    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    # Save all results into a single JSON
    save_logs(results, timestamp)


    # Save the best model automatically
    save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor)


    # evaluation!!!!!!!!!!!!!!!!!!!!
    

    # return
    return best_model_info
    


if __name__ == "__main__":
    best_model_info = main()




# Note. I might not have a best model if none of them is better than the recall threshold