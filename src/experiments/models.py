import os
import joblib
import json




# select best model
def select_best_model(results, models, recall_threshold=0.2):
    """
    Choose the best model according to recall + F1.
    Only here we attach the actual model object.
    """
    best_model_info = None
    best_f1 = -1.0
    best_model_recall = -1.0

    for model_name, info in results.items():
        recall = info["val_metrics"].get("recall", 0.0)
        f1 = info["val_metrics"].get("f1_score", 0.0)

        if recall >= recall_threshold and f1 > best_f1:
            best_f1 = f1
            best_model_recall = recall
            best_model_info = {
                "name": model_name,
                "model": models[model_name],  # attach model here
                "val_metrics": info["val_metrics"],
                "best_hyperparameters": info["best_hyperparameters"]
            }

    return best_model_info, best_model_recall, best_f1




# save best model
def save_best_model(best_model_info, timestamp, recall, best_f1, preprocessor=None):
    os.makedirs("outputs/saved_models", exist_ok=True)

    # Create experiment folder
    exp_dir = f"outputs/saved_models/experiment_{best_model_info['name']}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(exp_dir, "model.h5")
    best_model_info["model"].save(model_path)
    print(f"Model saved to {model_path}")

    # Save scaler (if exists)
    if preprocessor and getattr(preprocessor, "scaler", None) is not None:
        scaler_path = os.path.join(exp_dir, "scaler.pkl")
        joblib.dump(preprocessor.scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}")

    # Save encoder (if exists)
    if preprocessor and getattr(preprocessor, "encoder", None) is not None:
        encoder_path = os.path.join(exp_dir, "encoder.pkl")
        joblib.dump(preprocessor.encoder, encoder_path)
        print(f"Encoder saved to {encoder_path}")

    # Save feature order (columns)
    if preprocessor and getattr(preprocessor, "feature_columns", None) is not None:
        columns_path = os.path.join(exp_dir, "columns.json")
        with open(columns_path, "w") as f:
            json.dump(preprocessor.feature_columns, f, indent=4)
        print(f"Feature columns saved to {columns_path}")

    # Save small metadata file (optional but useful)
    meta_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "name": best_model_info["name"],
            "best_hyperparameters": best_model_info["best_hyperparameters"],
            "val_metrics": best_model_info["val_metrics"],
            "f1_score": best_f1,
            "recall": recall
        }, f, indent=4)
    print(f"Metadata saved to {meta_path}")

    print(f"\nBest model according to our criterion: {best_model_info['name']}, "
          f"F1={best_f1:.4f}, Recall={recall:.4f}")





