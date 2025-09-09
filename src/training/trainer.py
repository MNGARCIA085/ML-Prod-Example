from src.utils.io import make_run_dirs, log_message
from .utils import extract_optimizer_info
import json
import os
from src.utils.io import save_training_logs, log_message
from src.common.metrics import compute_f1


class ModelTrainer:
    """
    A utility class to train, evaluate, and log experiments with deep learning models.

    This class manages model training and evaluation while automatically handling:
    - Directory and log file creation
    - Tracking of hyperparameters and training history
    - Logging training and validation metrics (loss, accuracy, precision, recall, F1)

    Attributes
    ----------
    model_fn : callable
        A function that returns a compiled model instance when called.
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    model_name : str
        Name of the model (used for logging and saving).
    data_variant : str, optional
        Label for the data variant being trained on (default "default").
    log_dir : str, optional
        Directory where training logs and artifacts will be saved (default "logs/training").
    epochs : int, optional
        Number of training epochs (default 2).
    hyperparameters : dict, optional
        Dictionary of hyperparameters. If not provided, some will be inferred.
    callbacks : list, optional
        List of Keras callbacks (default empty list).
    """
    
    def __init__(self, model_fn, train_ds, val_ds, model_name, # pass train and val ds later to train
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


    # TRAINING
    def train(self):
        """
        Train the model, evaluate it, and save logs.

        Returns
        -------
        dict
            Dictionary with trained model, history, validation metrics, and hyperparameters.
        """
        log_message(f"=== Training: {self.model_name} ({self.data_variant}) ===", self.log_file)
        
        # Instanciate model
        self.model = self.model_fn()

        # Fill missing hyperparameters
        self._fill_missing_hyperparameters()

        # Train
        history = self._train(self.model, self.train_ds, self.val_ds, self.epochs, self.callbacks)

        # Evaluate
        val_metrics = self._evaluate_model(self.model, self.val_ds)


        # save logs
        save_training_logs(
            model_name=self.model_name,
            data_variant=self.data_variant,
            timestamp=self.timestamp,
            json_file=self.json_file,
            runs_index_file=self.runs_index_file,
            log_file=self.log_file,
            log_fn=lambda msg: log_message(msg, log_file=self.log_file),
            hyperparameters=self.hyperparameters,
            history=history,
            val_metrics=val_metrics
        )


        # return
        return {
            "model": self.model,
            "history": history,
            "val_loss": val_metrics['loss'],
            "val_accuracy": val_metrics['accuracy'],
            "val_precision": val_metrics['precision'],
            "val_recall": val_metrics['recall'],
            "val_f1": val_metrics['f1'],
            "hyperparameters": self.hyperparameters,
            "model_name": self.model_name,
            "data_variant": self.data_variant,
        }



    # ---------------- HELPER METHODS ----------------
    def _train(self, model, train_ds, val_ds, epochs, callbacks):
        """
        Internal training loop wrapper around model.fit.
        """
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
        )
        return history

    def _evaluate_model(self, model, val_ds):
        """
        Evaluate the model on the validation dataset.

        Returns
        -------
        dict
            Dictionary containing loss, accuracy, precision, recall, and F1 score.
        """
        val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)

        # Handle possible key name variations
        precision = val_metrics.get("precision") or val_metrics.get("precision_1") or 0.0
        recall = val_metrics.get("recall") or val_metrics.get("recall_1") or 0.0

        return {
            "loss": val_metrics.get("loss"),
            "accuracy": val_metrics.get("accuracy"),
            "precision": precision,
            "recall": recall,
            "f1": compute_f1(precision, recall),
        }

    def _fill_missing_hyperparameters(self):
        """
        Fill missing hyperparameters from the model (e.g., optimizer, learning rate).
        """

        # Extract optimizer info from model if missing
        if "optimizer" not in self.hyperparameters or "learning_rate" not in self.hyperparameters:
            opt_info = extract_optimizer_info(self.model)
            self.hyperparameters.setdefault("optimizer", opt_info["optimizer"])
            self.hyperparameters.setdefault("learning_rate", opt_info["learning_rate"])

        # Always store the number of epcohs
        self.hyperparameters["epochs"] = self.epochs





