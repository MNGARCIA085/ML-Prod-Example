import os
import json
import platform
import numpy as np
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
import time
from .utils import MetricsLogger,HistoryCapture,compute_f1,set_seed




# Model Tuner
class ModelTuner:
    def __init__(self, build_model_fn, log_dir="logs/tuning"):
        self.build_model_fn = build_model_fn

        # per-model folder
        self.model_dir = os.path.join(log_dir, build_model_fn.__name__)
        os.makedirs(self.model_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.model_dir, f"{build_model_fn.__name__}_tuner_{self.timestamp}.log")
        self.json_file = os.path.join(self.model_dir, f"{build_model_fn.__name__}_tuner_{self.timestamp}.json")

        # central index per model
        self.index_file = os.path.join(self.model_dir, "runs_index.json")
        if not os.path.exists(self.index_file):
            with open(self.index_file, "w") as f:
                json.dump([], f, indent=4)

    def log(self, msg):
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)

    def save_logs(self, model_name, best_hp, extra_hyperparams, history, val_metrics, elapsed_time=None):
        results = {
            "model_name": model_name,
            "timestamp": self.timestamp,
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
        with open(self.json_file, "w") as f:
            json.dump(results, f, indent=4)

        # Update central index
        with open(self.index_file, "r") as f:
            runs = json.load(f)

        runs.append({
            "timestamp": self.timestamp,
            "log_file": self.log_file,
            "json_file": self.json_file,
            "summary": {
                k: results.get(k) for k in ["val_metrics", "best_hyperparameters"]
            }
        })

        with open(self.index_file, "w") as f:
            json.dump(runs, f, indent=4)

        self.log(f"Logs saved to {self.log_file}")
        self.log(f"JSON metrics saved to {self.json_file}")



    def run(self, train_ds, val_ds, max_trials=2, executions_per_trial=1,
            epochs=2, patience=5, seed=42):
        
        # seed
        set_seed(seed)

        # start
        self.log(f"=== Starting tuner for {self.build_model_fn.__name__} ===")
        start_time = time.time()

        # tuning
        tuner = kt.RandomSearch(
            lambda hp: self.build_model_fn(hp),
            objective="val_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="tuner_results",
            project_name=self.build_model_fn.__name__,
            overwrite=True
        )

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        metrics_logger_cb = MetricsLogger(self.log)


        self.log("Starting tuner search...")
        tuner.search(train_ds, validation_data=val_ds, epochs=epochs,
                     callbacks=[early_stopping_cb, metrics_logger_cb])
        self.log("Tuner search finished.")

        
        # get best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.metric_names = best_model.metrics_names
        self.log(f"Best hyperparameters: {best_hp.values}")


        # retrain best model for full history
        history_cb = HistoryCapture()
        best_model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                       callbacks=[early_stopping_cb, metrics_logger_cb, history_cb], verbose=0)
        

        val_metrics = best_model.evaluate(val_ds, verbose=0)
        val_metrics_dict = {
                "loss": val_metrics[0],
                "accuracy": val_metrics[1],
                "precision": val_metrics[2],
                "recall": val_metrics[3],
                "f1_score": compute_f1(val_metrics[2], val_metrics[3])
        }

        # extra hyperparameters
        trained_epochs = len(history_cb.history["loss"])
        batch_size_tensor = getattr(train_ds, "_batch_size", None)
        batch_size = int(batch_size_tensor.numpy()) if batch_size_tensor is not None else "unknown"
        optimizer = type(best_model.optimizer).__name__
        learning_rate = float(tf.keras.backend.get_value(best_model.optimizer.learning_rate))

        extra_hyperparams = {
            "trained_epochs": trained_epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "final_learning_rate": learning_rate,
        }

        
        # elapsed time
        elapsed_time = time.time() - start_time

        self.save_logs(
            model_name=self.build_model_fn.__name__,
            best_hp=best_hp,
            extra_hyperparams=extra_hyperparams,
            history=history_cb.history,
            val_metrics=val_metrics_dict,
            elapsed_time=elapsed_time
        )

        self.log(f"=== End of tuner for {self.build_model_fn.__name__} ===")
        return best_model, best_hp, val_metrics_dict



"""
4️⃣ Recommended workflow

Run tuning for a model → checkpoint automatically saved per run.

Save logs (JSON) → include metrics, hyperparameters, and best_model_path.

Repeat for other models.

Load a specific checkpoint later using the best_model_path stored in your JSON index.
"""