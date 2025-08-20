import argparse
import json
import os
from datetime import datetime
from sklearn.metrics import f1_score
from src.data.preprocessor import BreastCancerPreprocessor
from src.models.model_dropout import build_model_with_dropout_tuner
from src.models.model_no_dropout import build_model_no_dropout_tuner
from src.models.baseline import baseline_with_tuner
from src.tuning.tuner import ModelTuner



# save results
def save_logs(results, timestamp):
    os.makedirs("logs/tuning", exist_ok=True)
    json_file = f"logs/tuning/all_models_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All models results saved to {json_file}")


# save best model (in a model format like .h5 and other info in a json file)
def save_best_model(best_model_info, timestamp, recall, best_f1):
    # Note I might not have a best model (one that its recall is lesse than the threshold)
    os.makedirs("saved_models", exist_ok=True)
    best_model_path = f"saved_models/best_model_{best_model_info['name']}_{timestamp}.h5"
    best_model_info["model"].save(best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Save JSON with best model info (hyperparameters + metrics)
    best_json_file = f"saved_models/best_model_info_{best_model_info['name']}_{timestamp}.json"
    with open(best_json_file, "w") as f:
        json.dump({
            "name": best_model_info["name"],
            "best_hyperparameters": best_model_info["best_hyperparameters"],
            "val_metrics": best_model_info["val_metrics"],
            "f1_score": best_f1
        }, f, indent=4)
    print(f"Best model info saved to {best_json_file}")

    print(f"\nBest model according to our criterion: {best_model_info['name']}, F1={best_f1:.4f}, Recall={recall:.4f}")
        # recall greater than a threshold and then greater f1-score



# get the best model and its data
def get_best_model(train_ds, val_ds, max_trials, epochs,models_to_tune, recall_threshold=0.2):
    results = {}
    best_model_info = None
    best_f1 = -1.0

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
                best_model_info = {
                    "name": model_name,
                    "model": best_model,
                    "val_metrics": val_metrics,
                    "best_hyperparameters": best_hp.values
                }

        print(f"{model_name} best hyperparameters: {best_hp.values}")
        print(f"{model_name} val metrics: {val_metrics}\n")

    return results, best_model_info, recall, best_f1





# MAIN
def main():
    parser = argparse.ArgumentParser(description="Tune multiple models with flags.")
    parser.add_argument("--baseline", action="store_true", help="Tune baseline model")
    parser.add_argument("--no_dropout", action="store_true", help="Tune model without dropout")
    parser.add_argument("--dropout", action="store_true", help="Tune model with dropout")
    parser.add_argument("--max_trials", type=int, default=3, help="Max trials for Keras Tuner")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per trial") #better default: 20
    args = parser.parse_args()

    tune_all = not (args.baseline or args.no_dropout or args.dropout)

    # Prepare data
    filepath = "data/data.csv"
    preprocessor = BreastCancerPreprocessor(batch_size=32)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

    models_to_tune = []
    if args.baseline or tune_all:
        models_to_tune.append(("baseline", baseline_with_tuner))
    if args.no_dropout or tune_all:
        models_to_tune.append(("no_dropout", build_model_no_dropout_tuner))
    if args.dropout or tune_all:
        models_to_tune.append(("dropout", build_model_with_dropout_tuner))


    # get best model
    results, best_model_info, recall, best_f1 = get_best_model(train_ds, val_ds, args.max_trials,args.epochs, models_to_tune)

    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all results into a single JSON
    save_logs(results, timestamp)

    # Save the best model automatically
    save_best_model(best_model_info, timestamp, recall, best_f1)
    

    # return
    return best_model_info
    


if __name__ == "__main__":
    best_model_info = main()




"""
best_model_info = choose_best_model(all_results, threshold=0.7)

if best_model_info is None:
    print("⚠️ No model met the threshold. Exiting...")
    return None
"""
