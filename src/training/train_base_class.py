import os
import json
from datetime import datetime
import tensorflow as tf

class ModelTrainer:
    def __init__(self, model_fn, train_ds, val_ds, test_ds, model_name, log_dir="logs", epochs=2, hyperparameters=None):
        self.model_fn = model_fn
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.model_name = model_name
        self.log_dir = log_dir
        self.epochs = epochs

        self.hyperparameters = hyperparameters or {}

        os.makedirs(self.log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"{self.model_name}_{self.timestamp}.log")
        self.json_file = os.path.join(self.log_dir, f"{self.model_name}_{self.timestamp}.json")

    def log(self, msg):
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)


    def extract_optimizer_info(self):
        opt = self.model.optimizer

        # Try these methods in order to get a meaningful optimizer name:
        if hasattr(opt, "_name"):
            opt_name = opt._name
        elif hasattr(opt, "get_config"):
            # Usually get_config contains 'name' key
            config = opt.get_config()
            opt_name = config.get("name", "unknown")
        else:
            # fallback to class name
            opt_name = opt.__class__.__name__

        try:
            lr = float(tf.keras.backend.get_value(opt.learning_rate))
        except Exception:
            lr = None

        return {"optimizer": opt_name, "learning_rate": lr}




    def train(self):
        self.log(f"=== Training session started: {self.model_name} ===")
        self.log(f"Timestamp: {self.timestamp}")

        self.model = self.model_fn() # buidls and compile a model


        # If hyperparameters dict missing optimizer info, extract from model
        if "optimizer" not in self.hyperparameters or "learning_rate" not in self.hyperparameters:
            opt_info = self.extract_optimizer_info()
            self.hyperparameters.setdefault("optimizer", opt_info["optimizer"])
            self.hyperparameters.setdefault("learning_rate", opt_info["learning_rate"])


        self.hyperparameters["epochs"] = self.epochs



        """
        self.model.compile( # im doing it twice this way; i can use previous fns.
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
        """

        self.history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.epochs
        )

        
        # Evaluate the final saved model on val and test sets
        self.log("=== Final evaluation on validation set ===")
        self.val_loss, self.val_acc, self.val_precision, self.val_recall = self.model.evaluate(self.val_ds)
        self.log(f"Validation Loss: {self.val_loss:.4f}, Validation Accuracy: {self.val_acc:.4f}")        
        self.val_f1 = 2 * (self.val_precision * self.val_recall) / (self.val_precision + self.val_recall + 1e-8)

        # use self to login (cehck!!!!!)





        self.log("=== Final evaluation on test set ===")
        self.test_loss, self.test_acc, self.test_precision, self.test_recall = self.model.evaluate(self.test_ds)
        self.log(f"Test Loss: {self.test_loss:.4f}, Test Accuracy: {self.test_acc:.4f}")

        self.save_logs()

        

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
                # + precison and recall!!!!! and f1 score
            },
        }

        with open(self.json_file, "w") as f:
            json.dump(results, f, indent=4)

        self.log(f"Logs saved to {self.log_file}")
        self.log(f"JSON metrics saved to {self.json_file}")


"""
trainer = ModelTrainer(model_fn=your_model_fn, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, model_name="model1")
model, history = trainer.train()
"""