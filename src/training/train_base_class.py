import os
import json
from datetime import datetime
import tensorflow as tf

class ModelTrainer:
    def __init__(self, model_fn, train_ds, val_ds, test_ds, model_name, log_dir="logs", epochs=5):
        self.model_fn = model_fn
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_name = model_name
        self.log_dir = log_dir
        self.epochs = epochs

        os.makedirs(self.log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{self.model_name}_{self.timestamp}.log")
        self.json_file = os.path.join(self.log_dir, f"{self.model_name}_{self.timestamp}.json")

    def log(self, msg):
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)

    def train(self):
        self.log(f"=== Training session started: {self.model_name} ===")
        self.log(f"Timestamp: {self.timestamp}")

        self.model = self.model_fn()
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        self.log("=== Final evaluation on test set ===")
        self.test_loss, self.test_acc = self.model.evaluate(self.test_ds)
        self.log(f"Test Loss: {self.test_loss:.4f}, Test Accuracy: {self.test_acc:.4f}")

        self.save_logs()

        return self.model, self.history

    def save_logs(self):
        results = {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_acc,
            "history": {
                "loss": list(map(float, self.history.history["loss"])),
                "val_loss": list(map(float, self.history.history["val_loss"])),
                "accuracy": list(map(float, self.history.history["accuracy"])),
                "val_accuracy": list(map(float, self.history.history["val_accuracy"]))
            }
        }

        with open(self.json_file, "w") as f:
            json.dump(results, f, indent=4)

        self.log(f"Logs saved to {self.log_file}")
        self.log(f"JSON metrics saved to {self.json_file}")


"""
trainer = ModelTrainer(model_fn=your_model_fn, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, model_name="model1")
model, history = trainer.train()
"""