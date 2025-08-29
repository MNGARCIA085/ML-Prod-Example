import os
import argparse
from src.evaluation.evaluator import load_data,evaluate 
from src.evaluation.utils import save_evaluation_metrics
from src.experiments.manager import get_last_experiment


# path where all experiments are stored
#BASE_DIR = "/home/marcos/Escritorio/AI-prod/ML-Prod-Example/outputs/saved_models"
BASE_DIR = "outputs/saved_models"



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



# main
def main():
    args = parse_args()

    if args.experiment:
        exp_path = os.path.join(BASE_DIR, args.experiment)
    elif args.last:
        exp_path = get_last_experiment(BASE_DIR)
    else:
        raise ValueError("Either --experiment or --last must be provided.")

    # evaluate
    experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc = evaluate(exp_path)

    # save to file
    save_evaluation_metrics(experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc)


if __name__ == "__main__":
    main()



