"""
from src.data.breast_cancer_preprocessor_normalized import BreastCancerPreprocessorNormalized
from src.models.model_dropout import build_model_with_dropout, build_model_with_dropout_tuner
from src.tuning.tuner_with_class import ModelTuner

def main():
    filepath = "data/breast_cancer_data.csv"

    # 1. Preprocess & create datasets
    preprocessor = BreastCancerPreprocessorNormalized(batch_size=32)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)
    X_train, _, _, _, _, _ = preprocessor.split_data(*preprocessor.preprocess_data())
    input_dim = X_train.shape[1]

    # 2. Train baseline model
    baseline_model = build_model_with_dropout(input_dim)
    baseline_model.fit(train_ds.concatenate(val_ds), epochs=10, validation_data=val_ds)
    val_perf = baseline_model.evaluate(val_ds)

    print(f"Baseline val accuracy: {val_perf[1]:.4f}")

    # 3. Hyperparameter tuning
    tuner = ModelTuner(build_model_fn=build_model_with_dropout_tuner)
    best_model, best_hp = tuner.run(train_ds, val_ds, max_trials=5, epochs=5)

    # 4. Retrain best tuned model on train+val
    best_model.fit(train_ds.concatenate(val_ds), epochs=10)

    # 5. Final evaluation on test set
    test_perf = best_model.evaluate(test_ds)
    print(f"Final test accuracy: {test_perf[1]:.4f}")

if __name__ == "__main__":
    main()
"""

