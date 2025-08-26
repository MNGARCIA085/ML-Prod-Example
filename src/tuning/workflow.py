from .tuner import ModelTuner
from src.experiments.models import select_best_model, save_best_model


# tune models
def tune_models(train_ds, val_ds, max_trials, epochs, models_to_tune):
    """
    Run tuning for each model and return JSON-serializable results 
    (no Keras models included).
    """
    results = {}
    models = {}  # keep actual models separately

    for model_name, build_fn in models_to_tune:
        print(f"\n=== Tuning {model_name} model ===")
        tuner = ModelTuner(build_model_fn=build_fn)
        best_model, best_hp, val_metrics = tuner.run(
            train_ds,
            val_ds,
            max_trials=max_trials,
            epochs=epochs
        )

        # JSON-safe results
        results[model_name] = {
            "best_hyperparameters": best_hp.values,
            "val_metrics": val_metrics,
        }

        # keep model in a separate dict
        models[model_name] = best_model

        print(f"{model_name} best hyperparameters: {best_hp.values}")
        print(f"{model_name} val metrics: {val_metrics}\n")

    return results, models




# it goes here and not under expermints/models because is specific to tuning
def get_best_model(train_ds, val_ds, max_trials, epochs, models_to_tune, recall_threshold=0.2):
    results, models = tune_models(train_ds, val_ds, max_trials, epochs, models_to_tune)
    best_model_info, best_model_recall, best_f1 = select_best_model(results, models, recall_threshold)
    return results, best_model_info, best_model_recall, best_f1
