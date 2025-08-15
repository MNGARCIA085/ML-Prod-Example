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


"""
great, now lets write the teh entire pipeline, tell me if my approach is ok: i start training 3 models with 2 diffenets data preproceossings; (total of 6 results); select the best 3 to make hyperarams tunning, and choose the nbest among them according to the f1-score; save that model
ChatGPT Plus

Yes — that’s a perfectly reasonable and standard experimental pipeline:

Baseline evaluation phase → run all model architectures with all preprocessing variants to see broad performance trends.

Narrow-down phase → pick the top performers (in your case, top 3 by validation F1) to save time and compute in tuning.

Hyperparameter tuning phase → optimize only those top candidates.

Final selection & saving → choose the single best tuned model by F1-score and save it for production/inference.

It’s essentially: coarse search → fine search → deploy.
The main advantage is that you don’t waste compute tuning models that clearly underperform early on.

Here’s a clean way to write the entire pipeline in one file, following that flow:


Why your plan works well

Two-phase search prevents over-tuning weak models.

F1-score as the main criterion is correct for imbalanced problems like breast cancer detection.

Saving all intermediate results means you can resume or re-analyze without rerunning everything.

The approach is scalable — adding more preprocessing pipelines or models is just adding items to a list.
"""