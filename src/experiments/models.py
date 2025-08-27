import os
import joblib
import json




# select the best model
def select_best_model(results, models, preprocessors_used, recall_threshold=0.2):
    """
    Choose the best (model + preprocessor) combination according to recall + F1.
    the greatest f1 of those whose recall is greater than a threshold
    """
    best_model_info = None
    best_f1 = -1.0
    best_model_recall = -1.0
    best_preprocessor = None

    for combo_name, info in results.items():
        recall = info["val_metrics"].get("recall", 0.0)
        f1 = info["val_metrics"].get("f1_score", 0.0)

        if recall >= recall_threshold and f1 > best_f1:
            best_f1 = f1
            best_model_recall = recall
            best_model_info = {
                "name": combo_name,
                "model": models[combo_name],
                "val_metrics": info["val_metrics"],
                "best_hyperparameters": info["best_hyperparameters"],
                "model_name": info["model_name"],
                "data_variant": info["data_variant"],
            }


            # best preprocessor
            best_preprocessor = preprocessors_used[combo_name]


    return best_model_info, best_preprocessor, best_model_recall, best_f1



# save best model
def save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor):
    """
    Save the best model along with its preprocessor artifacts and metadata.

    Parameters
    ----------
    best_model_info : dict
        Info returned by `select_best_model` containing model, name, metrics, etc.
    timestamp : str
        Unique identifier for this experiment run.
    recall : float
        Recall of the best model.
    best_f1 : float
        F1 score of the best model.
    preprocessor : object, optional
        Preprocessor instance used (if available). Should have
        attributes like scaler, encoder, feature_columns.
    """
    os.makedirs("outputs/saved_models", exist_ok=True)

    # Example: outputs/saved_models/experiment_baseline_simple_20250101-120000
    exp_dir = f"outputs/saved_models/experiment_{best_model_info['name']}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(exp_dir, "model.h5")
    best_model_info["model"].save(model_path)
    print(f"Model saved to {model_path}")

    # Save preprocessor artifacts if present
    if preprocessor and getattr(preprocessor, "scaler", None) is not None:
        scaler_path = os.path.join(exp_dir, "scaler.pkl")
        joblib.dump(preprocessor.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    if preprocessor and getattr(preprocessor, "encoder", None) is not None:
        encoder_path = os.path.join(exp_dir, "encoder.pkl")
        joblib.dump(preprocessor.encoder, encoder_path)
        print(f"Encoder saved to {encoder_path}")

    if preprocessor and getattr(preprocessor, "feature_columns", None) is not None:
        columns_path = os.path.join(exp_dir, "columns.json")
        with open(columns_path, "w") as f:
            json.dump(preprocessor.feature_columns, f, indent=4)
        print(f"Feature columns saved to {columns_path}")

    # Save metadata
    meta_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "name": best_model_info["name"],
            "model_name": best_model_info["model_name"],
            "data_variant": best_model_info["data_variant"],
            "best_hyperparameters": best_model_info["best_hyperparameters"],
            "val_metrics": best_model_info["val_metrics"],
            "f1_score": best_f1,
            "recall": recall
        }, f, indent=4)
    print(f"Metadata saved to {meta_path}")

    print(f"\nBest model according to our criterion: {best_model_info['name']} "
          f"(model={best_model_info['model_name']}, data={best_model_info['data_variant']}), "
          f"F1={best_f1:.4f}, Recall={recall:.4f}")



