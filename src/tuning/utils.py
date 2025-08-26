import random
import numpy as np
import os 
import json
import tensorflow as tf



# seed
def set_seed(seed=42):
    """Set seed for random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_fn):
        super().__init__()
        self.log_fn = log_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.log_fn(msg)


class HistoryCapture(tf.keras.callbacks.Callback):
    """Capture training history in a dict format."""
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))




# compute F1-score
def compute_f1(precision, recall):
    """Compute F1"""
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1


# save results
def save_logs(results, timestamp):
    os.makedirs("logs/tuning", exist_ok=True)
    json_file = f"logs/tuning/all_models_results_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"All models results saved to {json_file}")