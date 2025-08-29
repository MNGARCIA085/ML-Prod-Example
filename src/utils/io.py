import os
import json
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

def log_message(msg, log_file=None):
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")
    print(msg)
