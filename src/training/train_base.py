# src/training/train_base.py


from datetime import datetime

import os
import json
from datetime import datetime




def train_model(model_fn, train_ds, val_ds, test_ds, model_name, log_dir="logs"):
    """
    Base training function with structured logging.

    Parameters
    ----------
    model_fn : callable
        Function that returns a compiled Keras model.
    train_ds, val_ds, test_ds : tf.data.Dataset
        Datasets for training, validation, and testing.
    model_name : str
        Name of the model (used for logging).
    log_dir : str
        Directory for log files.
    """

    # --- Prepare logging paths ---
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
    json_file = os.path.join(log_dir, f"{model_name}_{timestamp}.json")

    def log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)

    log(f"=== Training session started: {model_name} ===")
    log(f"Timestamp: {timestamp}")

    # --- Build and compile model ---
    model = model_fn()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # --- Train ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    # --- Evaluate on test set ---
    log("=== Final evaluation on test set ===")
    test_loss, test_acc = model.evaluate(test_ds)
    log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # --- Save structured log ---
    results = {
        "model_name": model_name,
        "timestamp": timestamp,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "history": {
            "loss": list(map(float, history.history["loss"])),
            "val_loss": list(map(float, history.history["val_loss"])),
            "accuracy": list(map(float, history.history["accuracy"])),
            "val_accuracy": list(map(float, history.history["val_accuracy"]))
        }
    }

    with open(json_file, "w") as f:
        json.dump(results, f, indent=4)

    log(f"Logs saved to {log_file}")
    log(f"JSON metrics saved to {json_file}")

    return model, history, results

















def train_modelv0(model_fn, train_ds, val_ds, test_ds, model_name, log_dir="logs"):
    """
    Base training function.

    Parameters
    ----------
    Redo...........
    """

    
    # --- Build model ---
    model = model_fn()

    # --- Compile model ---
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # --- Logging setup ---
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    def log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)

    log(f"Starting training for model: {model_name}")

    # --- Train ---
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5
    )

    # --- Evaluate ---
    test_loss, test_acc = model.evaluate(test_ds)
    log(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return model, history


# with maybe custom compile fn. 

"""
def train_model(
    model_fn, 
    train_ds, val_ds, test_ds, model_name, 
    compile_fn=None, 
    custom_train_fn=None
):
    model = model_fn()

    if compile_fn:
        compile_fn(model)
    else:
        default_compile(model)

    if custom_train_fn:
        history = custom_train_fn(model, train_ds, val_ds)
    else:
        history = model.fit(train_ds, validation_data=val_ds, epochs=5)

    test_loss, test_acc = model.evaluate(test_ds)
    # logging etc.
    return model, history
"""