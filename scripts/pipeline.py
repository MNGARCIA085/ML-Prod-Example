import argparse
import json
from pathlib import Path
from datetime import datetime

from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.training.train_base_class3 import ModelTrainer
from src.models.baseline import build_compile_baseline
from src.models.model_dropout import build_compile_dropout
from src.models.model_no_dropout import build_compile_no_dropout


# ===== Mapping utilities =====
def get_preprocessor(name, batch_size):
    mapping = {
        "simple": BreastCancerPreprocessor(batch_size=batch_size),
        "normalized": BreastCancerPreprocessorNormalized(batch_size=batch_size),
    }
    if name not in mapping:
        raise ValueError(f"Unknown data variant: {name}")
    return mapping[name]


def get_model_fn(name, dropout_rate):
    mapping = {
        "baseline": build_compile_baseline,
        "no_dropout": build_compile_no_dropout,
        "dropout": lambda: build_compile_dropout(dropout_rate=dropout_rate)
    }
    return mapping[name]


# ===== Main pipeline =====
def main():
    filepath = 'data/data.csv'
    batch_size = 64
    dropout_rate = 0.2
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Train & tune models.")
    parser.add_argument("--data_variants", nargs="+", choices=["simple", "normalized"],
                        help="Data variants to use (default: all)")
    parser.add_argument("--models", nargs="+", choices=["baseline", "no_dropout", "dropout"],
                        help="Models to train (default: all)")
    args = parser.parse_args()

    data_variants = args.data_variants or ["simple", "normalized"]
    models_to_train = args.models or ["baseline", "no_dropout", "dropout"]

    experiments = []

    # ===== 1) Initial training phase =====
    for data_variant in data_variants:
        preprocessor = get_preprocessor(data_variant, batch_size)
        train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

        for model_name in models_to_train:
            print(f"Training model: {model_name} | data: {data_variant}")

            hyperparams = {"batch_size": batch_size}
            if model_name == "dropout":
                hyperparams["dropout_rate"] = dropout_rate

            # Prepare run folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = Path(f"logs/training/{model_name}/{data_variant}/{model_name}_{timestamp}")
            run_dir.mkdir(parents=True, exist_ok=True)

            trainer = ModelTrainer(
                model_fn=get_model_fn(model_name, dropout_rate),
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                model_name=model_name,
                data_variant=data_variant,
                hyperparameters=hyperparams
            )
            results = trainer.train()

            # Save model to file (not in results dict)
            model_path = run_dir / "model.h5"
            trainer.model.save(model_path)

            # Keep only serializable data in results
            clean_results = {
                "model_name": model_name,
                "data_variant": data_variant,
                "hyperparameters": hyperparams,
                "metrics": {k: v for k, v in results.items() if isinstance(v, (int, float, str))},
                "model_path": str(model_path),
                "run_dir": str(run_dir)
            }

            # Save metrics JSON in run folder
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(clean_results, f, indent=2)

            experiments.append(clean_results)

    # Save all initial results
    with open(results_dir / "initial_results.json", "w") as f:
        json.dump(experiments, f, indent=2)

    # ===== 2) Select top 3 by validation F1 =====
    top_3 = sorted(experiments, key=lambda x: x["metrics"]["val_f1"], reverse=True)[:3]
    print("\nTop 3 candidates for tuning:")
    for cand in top_3:
        print(f"{cand['model_name']} ({cand['data_variant']}): "
              f"F1={cand['metrics']['val_f1']:.4f}")

    # ===== 3) Hyperparameter tuning phase =====
    tuned_results = []
    for cand in top_3:
        print(f"\nTuning {cand['model_name']} ({cand['data_variant']})...")
        best_metrics, best_hyperparams, best_model_path = run_tuning(
            model_name=cand['model_name'],
            data_variant=cand['data_variant'],
            filepath=filepath,
            batch_size=batch_size
        )
        tuned_results.append({
            "model_name": cand['model_name'],
            "data_variant": cand['data_variant'],
            "metrics": best_metrics,
            "hyperparameters": best_hyperparams,
            "model_path": best_model_path
        })

    # Save tuned results
    with open(results_dir / "tuned_results.json", "w") as f:
        json.dump(tuned_results, f, indent=2)

    # ===== 4) Select final best model =====
    best_final = max(tuned_results, key=lambda x: x["metrics"]["val_f1"])
    print(f"\nBest final model: {best_final['model_name']} "
          f"({best_final['data_variant']}), F1={best_final['metrics']['val_f1']:.4f}")

    # ===== 5) Save final best model to "final_model" folder =====
    final_model_dir = results_dir / "final_model"
    final_model_dir.mkdir(exist_ok=True)
    Path(best_final["model_path"]).replace(final_model_dir / "model.h5")
    with open(final_model_dir / "metadata.json", "w") as f:
        json.dump(best_final, f, indent=2)
    print(f"Saved final model to {final_model_dir}")







# ===== Placeholder for tuning function =====
from datetime import datetime
import os
import json
from src.data.preprocessor import BreastCancerPreprocessor
from src.models.model_dropout import build_model_with_dropout_tuner
from src.models.model_no_dropout import build_model_no_dropout_tuner
from src.models.baseline import baseline_with_tuner
from src.tunning.tuner_with_class3 import ModelTuner
import numpy as np

# Utility to compute F1 from val metrics
def compute_f1_from_metrics(val_metrics):
    precision = val_metrics[2]
    recall = val_metrics[3]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

def run_tuning(model_name, data_variant, filepath, batch_size, max_trials=3, epochs=5):
    # Prepare data
    preprocessor = BreastCancerPreprocessor(batch_size=batch_size)  # you can extend for normalized
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

    # Map model name to tuner function
    model_fn_map = {
        "baseline": baseline_with_tuner,
        "no_dropout": build_model_no_dropout_tuner,
        "dropout": build_model_with_dropout_tuner
    }

    build_fn = model_fn_map[model_name]

    tuner = ModelTuner(build_model_fn=build_fn)
    best_model, best_hp, val_metrics, test_metrics = tuner.run(
        train_ds, val_ds, test_ds=test_ds,
        max_trials=max_trials, epochs=epochs
    )

    f1_val = compute_f1_from_metrics(val_metrics)

    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"saved_models/{model_name}_{data_variant}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.h5")
    best_model.save(model_path)

    # Save JSON info
    info_path = os.path.join(model_dir, "metadata.json")
    with open(info_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "data_variant": data_variant,
            "f1_val": f1_val,
            "val_metrics": val_metrics.tolist() if hasattr(val_metrics, "tolist") else val_metrics,
            "test_metrics": test_metrics.tolist() if hasattr(test_metrics, "tolist") else test_metrics,
            "hyperparameters": best_hp.values,
            "model_path": model_path
        }, f, indent=4)

    print(f"Saved best {model_name} model for {data_variant} at {model_path}")


    return (
        {"val_f1": f1_val, "val_metrics": val_metrics, "test_metrics": test_metrics},  # metrics dict
        best_hp.values,  # hyperparameters
        model_path       # model path
    )
    
    return {
        "model_name": model_name,
        "data_variant": data_variant,
        "f1_val": f1_val,
        "model_path": model_path,
        "hyperparameters": best_hp.values
    }



if __name__ == '__main__':
    main()
