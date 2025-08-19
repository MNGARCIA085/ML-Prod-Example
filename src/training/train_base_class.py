from .utils import get_early_stopping
from .utils import make_run_dirs
from .utils import extract_optimizer_info
from .utils import log_message
import json
import os



class ModelTrainer:
    def __init__(self, model_fn, train_ds, val_ds, model_name,
                 data_variant="default", log_dir="logs/training", epochs=2,
                 hyperparameters=None, callbacks=None):
        self.model_fn = model_fn
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.model_name = model_name
        self.data_variant = data_variant
        self.epochs = epochs
        self.hyperparameters = hyperparameters or {}
        self.callbacks = callbacks or []

        # Paths, timestamp, etc. (using your existing utils)
        self.run_dir, self.log_file, self.json_file, self.runs_index_file, self.timestamp = make_run_dirs(
            log_dir, model_name, data_variant
        )

    def log(self, msg):
        log_message(msg, self.log_file)

    def train(self):
        self.log(f"=== Training: {self.model_name} ({self.data_variant}) ===")
        
        # Instanciate model
        self.model = self.model_fn()

        # Fill missing hyperparameters
        if "optimizer" not in self.hyperparameters or "learning_rate" not in self.hyperparameters:
            opt_info = extract_optimizer_info(self.model)
            self.hyperparameters.setdefault("optimizer", opt_info["optimizer"])
            self.hyperparameters.setdefault("learning_rate", opt_info["learning_rate"])
        self.hyperparameters["epochs"] = self.epochs

        # Train with optional callbacks
        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs,
            callbacks=self.callbacks
        )

        # Evaluate
        self.val_loss, self.val_acc, self.val_precision, self.val_recall = self.model.evaluate(self.val_ds)
        self.val_f1 = 2 * (self.val_precision * self.val_recall) / (self.val_precision + self.val_recall + 1e-8)

        # save logs
        self.save_logs()
        
        # return
        return {
            "model": self.model,
            "history": self.history,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_acc,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1": self.val_f1,
            "hyperparameters": self.hyperparameters,
            "model_name": self.model_name,
            "data_variant": self.data_variant,
        }


    def save_logs(self):
        results = {
            "model_name": self.model_name,
            "data_variant": self.data_variant,
            "timestamp": self.timestamp,
            "hyperparameters": self.hyperparameters,
            "val_metrics":
                {
                    "loss": self.val_loss,
                    "accuracy": self.val_acc,
                    "precision": self.val_precision,
                    "recall": self.val_recall,
                    "f1": self.val_f1
                },
            "history": {k: list(map(float, v)) for k, v in self.history.history.items()}
        }

        with open(self.json_file, "w") as f:
            json.dump(results, f, indent=4)

        # Update runs index
        runs_index = []
        if os.path.exists(self.runs_index_file):
            with open(self.runs_index_file, "r") as f:
                runs_index = json.load(f)
        runs_index.append({"timestamp": self.timestamp, "json_file": os.path.basename(self.json_file)})
        with open(self.runs_index_file, "w") as f:
            json.dump(runs_index, f, indent=4)

        self.log(f"Logs saved to {self.log_file}")
        self.log(f"JSON saved to {self.json_file}")
        self.log(f"Run index updated: {self.runs_index_file}")