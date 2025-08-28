import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime
from typing import List, Dict


# for logs
def make_run_dirs(log_dir, model_name, data_variant):
    run_dir = os.path.join(log_dir, model_name, data_variant)
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(run_dir, f"{model_name}_{timestamp}.log")
    json_file = os.path.join(run_dir, f"{model_name}_{timestamp}.json")
    runs_index_file = os.path.join(run_dir, "runs_index.json")
    return run_dir, log_file, json_file, runs_index_file, timestamp

def log_message(msg, log_file=None):
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    print(msg)

# optimizer info
def extract_optimizer_info(model):
    opt = model.optimizer
    if hasattr(opt, "_name"):
        opt_name = opt._name
    elif hasattr(opt, "get_config"):
        config = opt.get_config()
        opt_name = config.get("name", "unknown")
    else:
        opt_name = opt.__class__.__name__
    try:
        lr = float(tf.keras.backend.get_value(opt.learning_rate))
    except Exception:
        lr = None
    return {"optimizer": opt_name, "learning_rate": lr}



# get best models according to our criterion
def get_top_n_models(results: List[Dict], recall_threshold: float = 0.2, top_n: int = 2):
    """
    Select top N models according to recall + F1.

    Args:
        results: List of result dicts (like your example).
        recall_threshold: Minimum recall required.
        top_n: Number of top models to return.

    Returns:
        List of top N result dicts sorted by F1 descending.
    """
    # Filter models that meet the recall threshold
    eligible = [r for r in results if r["val_recall"] >= recall_threshold]

    # Sort by F1 descending
    eligible_sorted = sorted(eligible, key=lambda r: r["val_f1"], reverse=True)

    # Return top N
    return eligible_sorted[:top_n]
