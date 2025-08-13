import os
import json
from datetime import datetime
import tensorflow as tf

class ModelTrainer:
    def __init__(self, model_fn, train_ds, val_ds, test_ds, model_name, log_dir="logs/training", epochs=2, hyperparameters=None):
        self.model_fn = model_fn
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_name = model_name
        self.epochs = epochs

        self.hyperparameters = hyperparameters or {}

        # Use per-model folder inside logs/training
        self.model_log_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.model_log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.model_log_dir, f"{self.model_name}_{self.timestamp}.log")
        self.json_file = os.path.join(self.model_log_dir, f"{self.model_name}_{self.timestamp}.json")

        # Initialize runs_index.json if it does not exist
        self.runs_index_file = os.path.join(self.model_log_dir, "runs_index.json")
        if not os.path.exists(self.runs_index_file):
            with open(self.runs_index_file, "w") as f:
                json.dump([], f)

    def log(self, msg):
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)

    def extract_optimizer_info(self):
        opt = self.model.optimizer
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

    def train(self):
        self.log(f"=== Training session started: {self.model_name} ===")
        self.log(f"Timestamp: {self.timestamp}")

        self.model = self.model_fn()

        if "optimizer" not in self.hyperparameters or "learning_rate" not in self.hyperparameters:
            opt_info = self.extract_optimizer_info()
            self.hyperparameters.setdefault("optimizer", opt_info["optimizer"])
            self.hyperparameters.setdefault("learning_rate", opt_info["learning_rate"])

        self.hyperparameters["epochs"] = self.epochs

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        # Evaluate validation set
        self.log("=== Final evaluation on validation set ===")
        self.val_loss, self.val_acc, self.val_precision, self.val_recall = self.model.evaluate(self.val_ds)
        self.val_f1 = 2 * (self.val_precision * self.val_recall) / (self.val_precision + self.val_recall + 1e-8)
        self.log(f"Validation Loss: {self.val_loss:.4f}, Accuracy: {self.val_acc:.4f}")

        # Evaluate test set
        self.log("=== Final evaluation on test set ===")
        self.test_loss, self.test_acc, self.test_precision, self.test_recall = self.model.evaluate(self.test_ds)
        self.log(f"Test Loss: {self.test_loss:.4f}, Accuracy: {self.test_acc:.4f}")

        self.save_logs()
        self.update_runs_index()

        return {
            "model": self.model,
            "history": self.history,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_acc,
            "val_precision": self.val_precision,
            "val_recall": self.val_recall,
            "val_f1": self.val_f1,
            "hyperparameters": self.hyperparameters,
        }

    def save_logs(self):
        results = {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_acc,
            "hyperparameters": self.hyperparameters,
            "history": {
                "loss": list(map(float, self.history.history["loss"])),
                "val_loss": list(map(float, self.history.history["val_loss"])),
                "accuracy": list(map(float, self.history.history["accuracy"])),
                "val_accuracy": list(map(float, self.history.history["val_accuracy"]))
            },
        }

        with open(self.json_file, "w") as f:
            json.dump(results, f, indent=4)

        self.log(f"Logs saved to {self.log_file}")
        self.log(f"JSON metrics saved to {self.json_file}")

    def update_runs_index(self):
        with open(self.runs_index_file, "r") as f:
            runs = json.load(f)

        runs.append({
            "timestamp": self.timestamp,
            "log_file": self.log_file,
            "json_file": self.json_file,
            "summary": {
                "val_loss": self.val_loss,
                "val_accuracy": self.val_acc,
                "val_precision": self.val_precision,
                "val_recall": self.val_recall,
                "val_f1": self.val_f1
            },
            "hyperparameters": self.hyperparameters
        })

        with open(self.runs_index_file, "w") as f:
            json.dump(runs, f, indent=4)
