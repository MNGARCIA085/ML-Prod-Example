import random
import numpy as np
import tensorflow as tf



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



# seed
def set_seed(seed=42):
    """Set seed for random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)