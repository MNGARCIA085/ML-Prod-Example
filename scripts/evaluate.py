import os
import argparse
import pandas as pd
import json
from datetime import datetime
from src.evaluation.evaluator import load_data,evaluate 

# path where all experiments are stored
BASE_DIR = "/home/marcos/Escritorio/AI-prod/ML-Prod-Example/outputs/saved_models"
OUTPUT_DIR = "/home/marcos/Escritorio/AI-prod/ML-Prod-Example/outputs/metrics"



# parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained experiment")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--experiment", type=str, help="Experiment folder name under saved_models/"
    )
    group.add_argument(
        "--last", action="store_true", help="Evaluate the latest experiment"
    )
    return parser.parse_args()


# get experiment
def get_last_experiment(base_dir: str) -> str:
    experiments = [f for f in os.listdir(base_dir) if f.startswith("experiment_")]
    if not experiments:
        raise ValueError("No experiments found in directory.")
    experiments.sort()  # timestamp in name ensures chronological order
    return os.path.join(base_dir, experiments[-1])


# save metrics
def save_metrics(experiment_path: str, eval_results, conf_matrix, fpr, tpr, thresholds, roc_auc):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exp_name = os.path.basename(experiment_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{exp_name}_{timestamp}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    metrics_dict = {
        "experiment": exp_name,
        "timestamp": timestamp,
        "evaluation_results": eval_results,
        "confusion_matrix": conf_matrix.tolist(),
        "fpr": fpr.tolist(),                        
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "roc_auc":roc_auc,
    }

    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Metrics saved to {filepath}")



# main
def main():
    args = parse_args()

    if args.experiment:
        exp_path = os.path.join(BASE_DIR, args.experiment)
    elif args.last:
        exp_path = get_last_experiment(BASE_DIR)
    else:
        raise ValueError("Either --experiment or --last must be provided.")

    experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc = evaluate(exp_path)

    # save to file
    save_metrics(experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc)


if __name__ == "__main__":
    main()



