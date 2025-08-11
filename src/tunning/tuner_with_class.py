import os
from datetime import datetime
import kerastuner as kt
import tensorflow as tf


# de momento estoy mezcalndo los logins





import os
import json
import csv
from datetime import datetime
import kerastuner as kt
import tensorflow as tf



class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_fn):
        super().__init__()
        self.log_fn = log_fn

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = f"Epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items())
        self.log_fn(msg)



class ModelTuner:
    def __init__(self, build_model_fn, log_dir="logs"):
        self.build_model_fn = build_model_fn
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{build_model_fn.__name__}_tuner_{timestamp}.log")

    def log(self, msg):
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)

    def run(self, train_ds, val_ds, max_trials=5, executions_per_trial=1, epochs=20, patience=3):
        self.log(f"=== Starting tuner for {self.build_model_fn.__name__} ===")


        def model_builder(hp):
            return self.build_model_fn(hp)

        tuner = kt.RandomSearch(
            model_builder,
            objective="val_accuracy",
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory="tuner_results",
            project_name=self.build_model_fn.__name__,
            overwrite=True,
        )

        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience, restore_best_weights=True
        )

        self.log("Starting tuner search...")


        metrics_logger_cb = MetricsLogger(self.log)

        tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=[early_stopping_cb, metrics_logger_cb]
        )
        self.log("Tuner search finished.")

        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]


        self.log("Final log:\n");

        self.log(f"Best hyperparameters: {best_hp.values}")
        self.log(f"=== End of tuner for {self.build_model_fn.__name__} ===")

        val_metrics = best_model.evaluate(val_ds, verbose=0)


        print(best_model.evaluate(val_ds))

        #test_metrics = None
        #if test_ds is not None:
        #    test_metrics = best_model.evaluate(test_ds, verbose=0)



        return best_model, best_hp, val_metrics  #, test_metrics


# -> elapsed time in logging