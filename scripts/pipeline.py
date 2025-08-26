
from src.training.trainer import ModelTrainer
from src.common.callbacks import get_callbacks
from src.models.baseline import build_compile_baseline
from src.models.model_dropout import build_compile_dropout
from src.models.model_no_dropout import build_compile_no_dropout
import argparse
from datetime import datetime
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.models.model_dropout import build_model_with_dropout_tuner
from src.models.model_no_dropout import build_model_no_dropout_tuner
from src.models.baseline import baseline_with_tuner
from src.tuning.tuner import ModelTuner
from src.config.constants import BREAST_CANCER_CSV_RAW
from src.evaluation.evaluator import evaluate
from src.evaluation.utils import save_evaluation_metrics
from src.experiments.manager import get_last_experiment
from src.tuning.workflow import get_best_model
from src.tuning.utils import save_logs
from src.experiments.models import save_best_model


# path where all experiments are stored
from src.config.constants import SAVED_MODELS_DIR





# MAIN
def main():
    
    # Simple setup
    models_to_train = ["baseline", "no_dropout", "dropout"]
    batch_size = 32 # maybe then change to args
    dropout_rate = 0.1



    # --------------- Data Loading and Preprocessing--------------------
    filepath = BREAST_CANCER_CSV_RAW
    preprocessor = BreastCancerPreprocessorNormalized(batch_size=32)
    train_ds, val_ds = preprocessor.get_datasets(filepath)



    # ----------------------- Training ---------------------------------

    experiments = []

    # Model function mapping
    model_fns = {
        "baseline": build_compile_baseline,
        "no_dropout": build_compile_no_dropout,
        "dropout": lambda: build_compile_dropout(dropout_rate=dropout_rate)
    }

    # callbacks
    callbacks = get_callbacks(
        use_early_stopping=True, 
        use_reduce_lr=False
    )


    for model_name in models_to_train:
        print(f"Training model: {model_name} | data: standardize")
        hyperparams = {"batch_size": batch_size}
        if model_name == "dropout":
            hyperparams["dropout_rate"] = dropout_rate

        trainer = ModelTrainer(
            model_fn=model_fns[model_name],
            train_ds=train_ds,
            val_ds=val_ds,
            model_name=model_name,
            data_variant='standardize',
            epochs=2,
            hyperparameters=hyperparams,
            callbacks=callbacks
        )
        results = trainer.train()
        experiments.append(results)



    #---------------Select 3 best models for hyperparameter tuning (models to tune)------------------
    # best f1 given the ones that passed recall test

    # to try hardcoded first
    models_to_tune = []
    models_to_tune.append(("baseline", baseline_with_tuner))
    models_to_tune.append(("dropout", build_model_with_dropout_tuner))



    #..................Hyperparameter tuning----------------------------------------
    # get best model (this is the fn. that tunes the model)
    max_trials = 1
    epochs = 5
    results, best_model_info, recall, best_f1 = get_best_model(train_ds, val_ds, max_trials, epochs, models_to_tune)

    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all results into a single JSON
    save_logs(results, timestamp) # change location later!!!!

    # Save the best model automatically
    save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor) # change location later!!!!!!!


    #.......................Evaluate last experminet...........................................
    
    # path to the last experiment
    exp_path = get_last_experiment(SAVED_MODELS_DIR)

    # evaluate
    experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc = evaluate(exp_path)

    # save to file
    save_evaluation_metrics(experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc)


if __name__ == '__main__':
    main()





