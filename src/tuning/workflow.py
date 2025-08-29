from .tuner import ModelTuner
from src.experiments.models import select_best_model, save_best_model



# tune combinations
def tune_combinations(
    preprocessors, models_to_tune, filepath, max_trials, epochs, allowed_combinations=None):
    """
    Tune combinations of model + preprocessor.

    Args:
        preprocessors (dict[str, object]): Mapping {name: preprocessor}.
        models_to_tune (dict[str, callable]): Mapping {model_name: build_fn}.
        filepath (str): Path to raw dataset.
        max_trials (int): Max number of tuning trials.
        epochs (int): Training epochs.
        allowed_combinations (set[tuple], optional): Only tune these (model_name, data_variant) pairs.
            If None, tune all combinations.

    Returns:
        dict: JSON-safe results keyed by (model_name, data_variant).
        dict: Best tuned models keyed by (model_name, data_variant).
        dict: Preprocessors used keyed by (model_name, data_variant).
    """
    results = {}
    models = {}
    preprocessors_used = {}

    for data_variant, preprocessor in preprocessors.items():
        train_ds, val_ds = preprocessor.get_datasets(filepath)

        for model_name, build_fn in models_to_tune.items():
            combo = (model_name, data_variant)

            # Skip combos not allowed
            if allowed_combinations and combo not in allowed_combinations:
                continue

            combo_name = f"{model_name}_{data_variant}"
            print(f"\n=== Tuning {combo_name} ===")

            # Run the tuner
            tuner = ModelTuner(build_model_fn=build_fn)
            best_model, best_hp, val_metrics = tuner.run(
                data_variant,
                train_ds,
                val_ds,
                max_trials=max_trials,
                epochs=epochs
            )

            # Save results
            results[combo_name] = {
                "best_hyperparameters": best_hp.values,
                "val_metrics": val_metrics,
                "data_variant": data_variant,
                "model_name": model_name,
            }
            models[combo_name] = best_model
            preprocessors_used[combo_name] = preprocessor

            print(f"{combo_name} best hyperparameters: {best_hp.values}")
            print(f"{combo_name} val metrics: {val_metrics}\n")

    return results, models, preprocessors_used



# it goes here and not under expermints/models because is specific to tuning
def get_best_model(preprocessors, models_to_tune, filepath, max_trials, epochs, recall_threshold=0.2, allowed_combinations=None):
    results, models, preprocessors_used = tune_combinations(preprocessors, models_to_tune, filepath, max_trials, epochs)
    best_model_info, best_preprocessor, best_model_recall, best_f1 = select_best_model(results, models, preprocessors_used, recall_threshold)
    return results, best_model_info, best_preprocessor, best_model_recall, best_f1


