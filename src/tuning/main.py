from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized

from src.data.preprocessor import BreastCancerPreprocessor

from src.models.model_dropout import build_model_with_dropout_tuner
from src.models.model_no_dropout import build_model_no_dropout_tuner
from src.tunning.tuner import run_tuner
from src.tunning.tuner_with_class2 import ModelTuner


def main():
    filepath = "data/data.csv"

    # Prepare data
    preprocessor = BreastCancerPreprocessor(batch_size=32)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)
    
    # Need input dimension for models
    # Assuming preprocessor exposes X_train internally, or add a getter for it:
    #X_train, _, _, _, _, _ = preprocessor.split_data(*preprocessor.preprocess_data())
    #input_dim = X_train.shape[1]

    tuner = ModelTuner(build_model_fn=build_model_no_dropout_tuner)
    best_model_a, best_hp_a, val_metrics_a, test_metrics_a = tuner.run(train_ds, val_ds, max_trials=1, epochs=2)

    # Evaluate best models on test set (optional)
    print("Model A evaluation on test set:", best_model_a.evaluate(test_ds))
    #print("Model B evaluation on test set:", best_model_b.evaluate(test_ds))

if __name__ == "__main__":
    main()


# python -m src.tunning main