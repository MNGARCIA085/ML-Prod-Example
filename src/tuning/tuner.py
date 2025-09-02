import os
import json
import platform
import numpy as np
import tensorflow as tf
import kerastuner as kt
from datetime import datetime
import time
from .utils import MetricsLogger,HistoryCapture,compute_f1,set_seed
from src.utils.io import save_tuning_logs, log_message



# Model Tuner
class ModelTuner:
    # constructor
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



    # run
    def run(self, data_variant, train_ds, val_ds, max_trials=2,
            executions_per_trial=1, epochs=2, patience=5, seed=42):

        set_seed(seed)

        # start
        log_message(f"=== Starting tuner for {self.build_model_fn.__name__} ===", self.log_file)
        start_time = time.time()

        tuner = self._create_tuner(max_trials, executions_per_trial)

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
        
        # metrics logger
        metrics_logger_cb = MetricsLogger(lambda msg: log_message(msg, log_file=self.log_file))

        # search
        log_message(f"Starting tuner search...", self.log_file)
        tuner.search(train_ds, validation_data=val_ds,
                     epochs=epochs, callbacks=[early_stopping_cb, metrics_logger_cb])
        log_message("Tuner search finished.", self.log_file)

        # Best model + hyperparameters
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.metric_names = best_model.metrics_names
        log_message(f"Best hyperparameters: {best_hp.values}",self.log_file)

        # Retrain
        history, trained_epochs = self._train_best_model(
            best_model, train_ds, val_ds, epochs, early_stopping_cb, metrics_logger_cb
        )

        # Evaluate
        val_metrics_dict = self._evaluate_model(best_model, val_ds)

        # Collect extra hyperparameters
        extra_hyperparams = self._collect_extra_hyperparams(best_model, train_ds, trained_epochs)

        elapsed_time = time.time() - start_time

        # Save logs
        save_tuning_logs(
            model_name=self.build_model_fn.__name__,
            data_variant=data_variant,
            timestamp=self.timestamp,
            json_file=self.json_file,
            index_file=self.index_file,
            log_file=self.log_file,
            log_fn=lambda msg: log_message(msg, log_file=self.log_file),   # still reuses your small log wrapper
            best_hp=best_hp,
            extra_hyperparams=extra_hyperparams,
            history=history,
            val_metrics=val_metrics_dict,
            elapsed_time=elapsed_time,
        )

        log_message(f"=== End of tuner for {self.build_model_fn.__name__} ===",self.log_file)

        # return
        return best_model, best_hp, val_metrics_dict


    # ---------------- HELPER METHODS ----------------

    def _create_tuner(self, max_trials, executions_per_trial):
        return kt.RandomSearch(
            lambda hp: self.build_model_fn(hp),
            objective="val_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="tuner_results",
            project_name=self.build_model_fn.__name__,
            overwrite=True
        )

    def _train_best_model(self, model, train_ds, val_ds, epochs, *callbacks):
        history_cb = HistoryCapture()
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[*callbacks, history_cb],
            verbose=0
        )
        trained_epochs = len(history_cb.history.get("loss", []))
        return history_cb.history, trained_epochs

    def _evaluate_model(self, model, val_ds):
        val_metrics = model.evaluate(val_ds, verbose=0)
        return {
            "loss": val_metrics[0],
            "accuracy": val_metrics[1],
            "precision": val_metrics[2],
            "recall": val_metrics[3],
            "f1_score": compute_f1(val_metrics[2], val_metrics[3])
        }
        """
        val_metrics_dict = {name: value for name, value in zip(model.metrics_names, val_metrics)}
        """

    def _collect_extra_hyperparams(self, model, train_ds, trained_epochs):
        batch_size_tensor = getattr(train_ds, "_batch_size", None)
        batch_size = int(batch_size_tensor.numpy()) if batch_size_tensor is not None else "unknown"
        optimizer = type(model.optimizer).__name__
        learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))

        return {
            "trained_epochs": trained_epochs,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "final_learning_rate": learning_rate,
        }









"""
4️⃣ Recommended workflow

Run tuning for a model → checkpoint automatically saved per run.

Save logs (JSON) → include metrics, hyperparameters, and best_model_path.

Repeat for other models.

Load a specific checkpoint later using the best_model_path stored in your JSON index.
"""