import argparse
import json
from datetime import datetime
from sklearn.metrics import f1_score
from src.data.preprocessor import BreastCancerPreprocessor
from src.models.model_dropout import build_model_with_dropout_tuner
from src.models.model_no_dropout import build_model_no_dropout_tuner
from src.models.baseline import baseline_with_tuner  # if exists
from src.tunning.tuner_with_class3 import ModelTuner # 2ok



# que vaya a utils después!!!!!
def compute_f1_from_metrics(val_metrics):
    """Compute F1 from val_metrics list assuming precision=val_metrics[2], recall=val_metrics[3]"""
    precision = val_metrics[2]
    recall = val_metrics[3]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


def main():
    parser = argparse.ArgumentParser(description="Tune multiple models with flags.")
    parser.add_argument("--baseline", action="store_true", help="Tune baseline model")
    parser.add_argument("--no_dropout", action="store_true", help="Tune model without dropout")
    parser.add_argument("--dropout", action="store_true", help="Tune model with dropout")
    parser.add_argument("--max_trials", type=int, default=3, help="Max trials for Keras Tuner")
    parser.add_argument("--epochs", type=int, default=5, help="Max epochs per trial") #20
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

    results = {}
    best_model_info = None
    best_f1 = -1.0

    for model_name, build_fn in models_to_tune:
        print(f"\n=== Tuning {model_name} model ===")
        tuner = ModelTuner(build_model_fn=build_fn)
        best_model, best_hp, val_metrics, test_metrics = tuner.run(
            train_ds,
            val_ds,
            test_ds=test_ds,
            max_trials=args.max_trials,
            epochs=args.epochs
        )

        results[model_name] = {
            "best_hyperparameters": best_hp.values,
            "val_metrics": val_metrics.tolist() if hasattr(val_metrics, "tolist") else val_metrics,
            "test_metrics": test_metrics.tolist() if test_metrics is not None and hasattr(test_metrics, "tolist") else test_metrics
        }

        # compute F1 for this model
        f1 = compute_f1_from_metrics(val_metrics)
        if f1 > best_f1:
            best_f1 = f1
            best_model_info = {
                "name": model_name,
                "model": best_model,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "best_hyperparameters": best_hp.values
            }

        print(f"{model_name} best hyperparameters: {best_hp.values}")
        print(f"{model_name} evaluation on test set: {best_model.evaluate(test_ds)}\n")

    # Save all results into a single JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"logs/tuning/all_models_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All models results saved to {json_file}")

    print(f"\nBest model according to F1-score: {best_model_info['name']}, F1={best_f1:.4f}")

    # best model according to f1-score and stability
    # https://chatgpt.com/c/689e2a31-4828-832e-aac8-fb344e05fcdc

    return best_model_info


    # maybe save the best model (to deploy it later!!!!!!!)
    """
    import os

    os.makedirs("saved_models", exist_ok=True)
    best_model.save("saved_models/best_model.h5")

    # Save hyperparameters & metrics
    import json

    with open("saved_models/best_model_info.json", "w") as f:
        json.dump({
            "best_hyperparameters": best_model_info["best_hyperparameters"],
            "val_metrics": best_model_info["val_metrics"],
            "test_metrics": best_model_info["test_metrics"]
        }, f, indent=4)
    """

    """
    better
    # Save all results into a single JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs/tuning", exist_ok=True)
    json_file = f"logs/tuning/all_models_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All models results saved to {json_file}")

    # Save the best model automatically
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
            "test_metrics": best_model_info["test_metrics"],
            "f1_score": best_f1
        }, f, indent=4)
    print(f"Best model info saved to {best_json_file}")

    print(f"\nBest model according to F1-score: {best_model_info['name']}, F1={best_f1:.4f}")
    return best_model_info
    """


if __name__ == "__main__":
    best_model_info = main()




"""
chossing bets model according to f1 and stability
import numpy as np

def stability_aware_f1_score(f1_history, epsilon=1e-8):
    
    #Compute a combined score: mean F1 weighted by stability.
    
    #f1_history: list or array of F1 scores per epoch
    
    

    f1_history = np.array(f1_history)
    mean_f1 = np.mean(f1_history)
    std_f1 = np.std(f1_history)
    
    stability = 1 - (std_f1 / (mean_f1 + epsilon))
    score = mean_f1 * stability
    return score, mean_f1, stability

# Example usage
f1_vals = [0.78, 0.79, 0.81, 0.80, 0.79]
score, mean_f1, stab = stability_aware_f1_score(f1_vals)

print(f"Combined Score: {score:.3f}, Mean F1: {mean_f1:.3f}, Stability: {stab:.3f}")
"""
