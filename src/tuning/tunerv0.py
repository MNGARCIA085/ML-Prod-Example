import keras_tuner as kt
import tensorflow as tf



import os
from datetime import datetime

def run_tuner(build_model_fn, train_ds, val_ds, max_trials=5, executions_per_trial=1, epochs=20, patience=3):
    """
    Runs hyperparameter tuning with early stopping and pruning.

    Args:
        build_model_fn: function(hp) returning compiled tf.keras.Model
        train_ds: tf.data.Dataset for training
        val_ds: tf.data.Dataset for validation
        max_trials: max hyperparameter trials
        executions_per_trial: executions per trial (for robustness)
        epochs: max epochs per trial
        patience: epochs with no improvement before early stopping

    Returns:
        best_model: best found model (compiled)
        best_hp: best hyperparameters object
    """

    # Setup logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{build_model_fn.__name__}_tuner_{timestamp}.log")

    def log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        print(msg)

    log(f"=== Starting tuner for {build_model_fn.__name__} at {timestamp} ===")

    def model_builder(hp):
        return build_model_fn(hp)

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="tuner_results",
        project_name=build_model_fn.__name__,
        overwrite=True,
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True
    )

    log("Starting tuner search...")
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping_cb]
    )
    log("Tuner search finished.")

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    log(f"Best hyperparameters: {best_hp.values}")
    log(f"=== End of tuner for {build_model_fn.__name__} ===")



    return best_model, best_hp




def run_tunervo(build_model_fn, #input_dim 
                train_ds, val_ds, 
              max_trials=5, executions_per_trial=1, 
              epochs=20, patience=3):
    """
    Runs hyperparameter tuning with early stopping and pruning.

    Args:
        build_model_fn: function(hp, input_dim) returning compiled tf.keras.Model
        input_dim: int, input feature dimension
        train_ds: tf.data.Dataset for training
        val_ds: tf.data.Dataset for validation
        max_trials: max hyperparameter trials
        executions_per_trial: executions per trial (for robustness)
        epochs: max epochs per trial
        patience: epochs with no improvement before early stopping

    Returns:
        best_model: best found model (compiled)
        best_hp: best hyperparameters object
    """
    def model_builder(hp):
        return build_model_fn(hp)#, input_dim)

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_accuracy",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="tuner_results",
        project_name=build_model_fn.__name__,
        overwrite=True,
    )

    # Early stopping callback to avoid wasting epochs on bad trials
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience, restore_best_weights=True
    )

    # Keras Tuner’s built-in pruning callback (works with RandomSearch and others)
    #tuner_pruning_cb = kt.callbacks.TunerCallback(tuner)

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping_cb] #, tuner_pruning_cb]
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Best hyperparameters for {build_model_fn.__name__}: {best_hp.values}")
    return best_model, best_hp
