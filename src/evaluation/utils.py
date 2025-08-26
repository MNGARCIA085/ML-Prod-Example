import os
import json
from datetime import datetime
from src.config.constants import OUTPUT_DIR


# save evaluation metrics
def save_evaluation_metrics(experiment_path: str, eval_results, conf_matrix, fpr, tpr, thresholds, roc_auc):
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