
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
from src.training.experiment_runner import run_experiment
from src.models.factory import get_model_fns
from src.data.utils import get_preprocessors

# path where all experiments are stored
from src.config.constants import SAVED_MODELS_DIR




# MAIN
def main():
    
    # Simple setup
    models_to_train = ["baseline", "no_dropout", "dropout"]
    batch_size = 32 # maybe then change to args
    dropout_rate = 0.1
    epochs = 2



    # --------------- Data Loading and Preprocessing--------------------
    filepath = BREAST_CANCER_CSV_RAW
    preprocessor = BreastCancerPreprocessorNormalized(batch_size=batch_size)
    train_ds, val_ds = preprocessor.get_datasets(filepath)
    # quizás luego no xq lo saco de prepreocessors



    # ----------------------- Training ---------------------------------

    # preprocessors
    data_variants = ["simple", "standardize"] # maybe only standrazie later
    preprocessors = get_preprocessors(data_variants, batch_size)
    
    # Model function mapping
    model_fns = get_model_fns(dropout_rate)

    # callbacks
    callbacks = get_callbacks(
        use_early_stopping=True, 
        use_reduce_lr=False
    )

    # run experiments
    experiments = run_experiment(model_fns,
                             preprocessors, filepath,
                             epochs=epochs, callbacks=callbacks,
                             batch_size=batch_size,
                             dropout_rate=dropout_rate)


    

    #---------------Select 3 best models for hyperparameter tuning (models to tune)------------------
    # best f1 given the ones that passed recall test

    # to try hardcoded first
    models_to_tune = []
    models_to_tune.append(("baseline", baseline_with_tuner))
    models_to_tune.append(("dropout", build_model_with_dropout_tuner))



    #---------------------------------Hyperparameter tuning-------------------------------------------
    
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





