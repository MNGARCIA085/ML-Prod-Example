import os
import json
import platform
import tensorflow as tf
from datetime import datetime


# for logs
def make_run_dirs(log_dir, model_name, data_variant):
    run_dir = os.path.join(log_dir, model_name, data_variant)
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(run_dir, f"{model_name}_{timestamp}.log")
    json_file = os.path.join(run_dir, f"{model_name}_{timestamp}.json")
    runs_index_file = os.path.join(run_dir, "runs_index.json")
    return run_dir, log_file, json_file, runs_index_file, timestamp


# log message
def log_message(msg, log_file=None, to_console=True):
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    if to_console:
        print(msg)


# training logs
def save_training_logs(
    model_name,
    data_variant,
    timestamp,
    json_file,
    runs_index_file,
    log_file,
    log_fn,
    hyperparameters,
    history,
    val_metrics
):
    results = {
        "model_name": model_name,
        "data_variant": data_variant,
        "timestamp": timestamp,
        "hyperparameters": hyperparameters,
        "val_metrics": val_metrics,
        "history": {k: list(map(float, v)) for k, v in history.history.items()}
    }

    # Save JSON
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)

    # Update central index
    runs_index = []
    if os.path.exists(runs_index_file):
        with open(runs_index_file, "r") as f:
            runs_index = json.load(f)

    runs_index.append({"timestamp": timestamp, "json_file": os.path.basename(json_file)})

    with open(runs_index_file, "w") as f:
        json.dump(runs_index, f, indent=4)

    # Log messages
    log_fn(f"Logs saved to {log_file}")
    log_fn(f"JSON saved to {json_file}")
    log_fn(f"Run index updated: {runs_index_file}")



# tuning logs
def save_tuning_logs(
    model_name,
    data_variant,
    timestamp,
    json_file,
    index_file,
    log_file,
    log_fn,
    best_hp,
    extra_hyperparams,
    history,
    val_metrics,
    elapsed_time=None,
):
    results = {
        "model_name": model_name,
        "data_variant": data_variant,
        "timestamp": timestamp,
        "elapsed_time_sec": elapsed_time,
        "best_hyperparameters": {**best_hp.values, **extra_hyperparams},
        "environment": {
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "keras_version": tf.keras.__version__,
            "os": platform.platform(),
        },
        "history": history,
        "val_metrics": val_metrics,
    }

    # Save JSON log
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)

    # Update central index
    runs = []
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            runs = json.load(f)

    runs.append({
        "timestamp": timestamp,
        "log_file": log_file,
        "json_file": json_file,
    })

    with open(index_file, "w") as f:
        json.dump(runs, f, indent=4)

    log_fn(f"Logs saved to {log_file}")
    log_fn(f"JSON metrics saved to {json_file}")



# save results for tuning
def save_tuning_all(results, timestamp):
    os.makedirs("logs/tuning", exist_ok=True)
    json_file = f"logs/tuning/all_models_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All models results saved to {json_file}")


