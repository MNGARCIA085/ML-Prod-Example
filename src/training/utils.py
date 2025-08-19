import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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



# callbacks
def get_early_stopping(monitor='val_loss', patience=5, restore_best_weights=True):
    """Return a configured EarlyStopping callback."""
    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights)

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