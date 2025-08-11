from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized

from src.data.preprocessor import BreastCancerPreprocessor

from src.models.model_dropout import build_model_with_dropout_tuner
from src.tunning.tuner import run_tuner



def main():
    filepath = "data/data.csv"



    # Prepare data
    preprocessor = BreastCancerPreprocessor(batch_size=32)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)
    
    # Need input dimension for models
    # Assuming preprocessor exposes X_train internally, or add a getter for it:
    #X_train, _, _, _, _, _ = preprocessor.split_data(*preprocessor.preprocess_data())
    #input_dim = X_train.shape[1]



    # Tune Model A
    #best_model_a, best_hp_a = tuner.run_tuner(model_a.build_model_a, input_dim, train_ds, val_ds)
    

    # ok!!!!!!!!!!!!!
    #best_model, best_hp = run_tuner(build_model_with_dropout_tuner, train_ds, val_ds, max_trials=5, epochs=5)
    #best_model_a = best_model

    # with a class
    from src.tunning.tuner_with_class import ModelTuner
    tuner = ModelTuner(build_model_fn=build_model_with_dropout_tuner)
    best_model, best_hp, val_metrics = tuner.run(train_ds, val_ds, max_trials=1, epochs=5)
    print('dsfdsfdsfdds',val_metrics)
    best_model_a = best_model



    # Tune Model B
    #best_model_b, best_hp_b = tuner.run_tuner(model_b.build_model_b, input_dim, train_ds, val_ds)

    # Evaluate best models on test set (optional)
    print("Model A evaluation on test set:", best_model_a.evaluate(test_ds))
    #print("Model B evaluation on test set:", best_model_b.evaluate(test_ds))

if __name__ == "__main__":
    main()
