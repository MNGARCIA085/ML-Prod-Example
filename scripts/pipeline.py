import argparse
from datetime import datetime

# Constants
from src.config.constants import BREAST_CANCER_CSV_RAW, SAVED_MODELS_DIR

# Callbacks
from src.common.callbacks import get_callbacks

# Data
from src.data.utils import get_preprocessors

# Models
from src.models.factory import get_model_fns, get_model_fns_tuner

# Training
from src.training.experiment_runner import run_experiment
from src.training.utils import get_top_n_models

# Experiments
from src.experiments.manager import get_last_experiment
from src.experiments.models import save_best_model

# Tuning
from src.tuning.utils import save_logs
from src.tuning.workflow import get_best_model

# Evaluation
from src.evaluation.evaluator import evaluate
from src.evaluation.utils import save_evaluation_metrics






# MAIN
def main():
    
    # Simple setup
    models_to_train = ["baseline", "no_dropout", "dropout"]
    data_variants = ["simple", "standardize"]
    batch_size = 32 # maybe then change to args
    dropout_rate = 0.1
    epochs = 2
    recall_threshold = 0.8
    top_n = 3



    # --------------- Data Path and Preprocessors--------------------------
    
    # Path to raw data
    filepath = BREAST_CANCER_CSV_RAW

    # preprocessors
    preprocessors = get_preprocessors(data_variants, batch_size)


    # ----------------------- Training ---------------------------------
  
    # Model function mapping
    model_fns = get_model_fns(dropout_rate)

    # callbacks
    callbacks = get_callbacks(
        use_early_stopping=True, 
        use_reduce_lr=False
    )

    # run experiments
    results = run_experiment(model_fns,
                             preprocessors, filepath,
                             epochs=epochs, callbacks=callbacks,
                             batch_size=batch_size,
                             dropout_rate=dropout_rate)


    

    #---------------Select 3 best models for hyperparameter tuning (models to tune)------------------
    
    # select best experiments
    top_models = get_top_n_models(results, recall_threshold=recall_threshold, top_n=top_n)


    # allowed comibations to tune (only the best ones)
    allowed_combinations = []
    for m in top_models:
        allowed_combinations.append((m['model_name'], m['data_variant']))

    #---------------------------------Hyperparameter tuning-------------------------------------------

    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Models (with tunen)
    model_fns = get_model_fns_tuner()

    # get best (model + prep)
    results, best_model_info, preprocessor, recall, best_f1 = get_best_model(preprocessors, model_fns, filepath, 
                                                        max_trials=1, epochs=epochs, recall_threshold=recall_threshold,
                                                        allowed_combinations=allowed_combinations)


    # Save the best model automatically
    save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor)

    # Save all results into a single JSON
    save_logs(results, timestamp)



    #.......................Evaluate last experminet...........................................
    
    # path to the last experiment
    exp_path = get_last_experiment(SAVED_MODELS_DIR)

    # evaluate
    experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc = evaluate(exp_path)

    # save to file
    save_evaluation_metrics(experiment_path, res, conf_matrix, fpr, tpr, thresholds, roc_auc)


if __name__ == '__main__':
    main()





