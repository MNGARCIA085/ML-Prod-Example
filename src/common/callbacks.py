import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau



# Metrics logger
class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_fn):
        super().__init__()
        self.log_fn = log_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.log_fn(msg)


# Capture history
class HistoryCapture(tf.keras.callbacks.Callback):
    """Capture training history in a dict format."""
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))


# Early Stopping
def get_early_stopping(monitor='val_loss', patience=5, restore_best_weights=True):
    """Return a configured EarlyStopping callback."""
    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights)


# Reduce LR on Plateau
def get_reduce_lr_on_plateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6):
    """Return a ReduceLROnPlateau callback."""
    return ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, min_lr=min_lr)


# get all callbacks
def get_callbacks(use_early_stopping=True, use_reduce_lr=False):
    """Return a list of callbacks based on flags."""
    callbacks = []
    if use_early_stopping:
        callbacks.append(get_early_stopping())
    if use_reduce_lr:
        callbacks.append(get_reduce_lr_on_plateau())
    return callbacks