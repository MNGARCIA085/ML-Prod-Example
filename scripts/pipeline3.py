import argparse
import os
import json
from datetime import datetime
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.training.train_base_class3 import ModelTrainer
from src.tunning.tuner_with_class3 import ModelTuner
from src.models.baseline import build_compile_baseline
from src.models.model_dropout import build_compile_dropout
from src.models.model_no_dropout import build_compile_no_dropout






import subprocess

# Optional: run tests before training
def run_tests():
    print("Running pytest for data prep and models...")
    result = subprocess.run(["pytest", "-q", "--tb=short", "tests/"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Some tests failed! Aborting pipeline.")
        exit(1)



# Utility to compute F1 from metrics
def compute_f1_from_metrics(metrics):
    precision = metrics[2]
    recall = metrics[3]
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1




# main
def main():
    parser = argparse.ArgumentParser(description="Full training + tuning pipeline")
    parser.add_argument("--max_trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--run_tests", action="store_true", help="Run pytest tests before training")

    args = parser.parse_args()

    if args.run_tests:
        run_tests()  # call your test-running function




    filepath = "data/data.csv"
    batch_size = 64

    # --- Define preprocessing variants ---
    preprocessors = [
        ("standard", BreastCancerPreprocessor(batch_size=batch_size)),
        ("normalized", BreastCancerPreprocessorNormalized(batch_size=batch_size))
    ]

    # --- Define models ---
    models = [
        ("baseline", build_compile_baseline),
        ("no_dropout", build_compile_no_dropout),
        ("dropout", lambda: build_compile_dropout(dropout_rate=0.2))
    ]

    # --- Step 1: Train all model + preprocessing combinations ---
    print("=== Training all model + preprocessing variants ===")
    experiments = []
    for data_variant, preprocessor in preprocessors:
        train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)
        for model_name, model_fn in models:
            print(f"\nTraining {model_name} on {data_variant} data")
            trainer = ModelTrainer(
                model_fn=model_fn,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                model_name=model_name,
                data_variant=data_variant,
                epochs=args.epochs
            )
            result = trainer.train()
            experiments.append(result)

    # --- Step 2: Select top 3 models by val_f1 ---
    top_candidates = sorted(experiments, key=lambda x: x["val_f1"], reverse=True)[:3]

    # --- Step 3: Hyperparameter tuning for top candidates ---
    print("\n=== Hyperparameter tuning for top candidates ===")
    tuned_results = []
    for cand in top_candidates:
        print(f"\nTuning {cand['model_name']} with {cand['data_variant']} data")
        # Select proper build_fn
        if cand["model_name"] == "baseline":
            from src.models.baseline import baseline_with_tuner as build_fn
        elif cand["model_name"] == "no_dropout":
            from src.models.model_no_dropout import build_model_no_dropout_tuner as build_fn
        elif cand["model_name"] == "dropout":
            from src.models.model_dropout import build_model_with_dropout_tuner as build_fn

        # run tuner
        tuner = ModelTuner(build_model_fn=build_fn)
        best_model, best_hp, val_metrics, test_metrics = tuner.run(
            train_ds=preprocessor.get_datasets(filepath)[0],
            val_ds=preprocessor.get_datasets(filepath)[1],
            test_ds=preprocessor.get_datasets(filepath)[2],
            max_trials=args.max_trials,
            epochs=args.epochs
        )

        f1_val = compute_f1_from_metrics(val_metrics)
        tuned_results.append({
            "model_name": cand["model_name"],
            "data_variant": cand["data_variant"],
            "f1_val": f1_val,
            "best_model": best_model,
            "best_hp": best_hp.values,
            "timestamp": tuner.timestamp
        })

    # --- Step 4: Select final best model ---
    final_best = max(tuned_results, key=lambda x: x["f1_val"])

    # --- Step 5: Save final best model + metadata ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_dir = f"saved_models/{final_best['model_name']}_{final_best['data_variant']}_{timestamp}"
    os.makedirs(final_dir, exist_ok=True)
    model_path = os.path.join(final_dir, "best_model.h5")
    final_best["best_model"].save(model_path)

    metadata_path = os.path.join(final_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "model_name": final_best["model_name"],
            "data_variant": final_best["data_variant"],
            "f1_val": final_best["f1_val"],
            "best_hp": final_best["best_hp"],
            "model_path": model_path,
            "timestamp": final_best["timestamp"]
        }, f, indent=4)

    print(f"\n=== Final best model saved ===")
    print(f"Path: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"F1-score: {final_best['f1_val']:.4f}")

if __name__ == "__main__":
    main()


"""
about results
def main():
    print("⚠ Note: Using the Wisconsin Breast Cancer dataset (569 samples, clean features).")
    print("This dataset is small and linearly separable, so near-perfect metrics are expected.")
    print("The purpose here is methodology (pipeline, preprocessing, tuning), not model performance.\n")
    
    filepath = 'data/data.csv'
    # ... rest of your code
self.log("⚠ Dataset: Wisconsin Breast Cancer (small & clean).")
self.log("Purpose: pipeline methodology demonstration; high metrics are expected.\n")


results = {
    "model_name": self.model_name,
    "data_variant": self.data_variant,
    "timestamp": self.timestamp,
    "note": "Dataset is small and clean; metrics near-perfect as expected; focus on methodology.",
    ...
}
"""