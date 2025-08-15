from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized

#from src.training.train_base import train_model


from src.training.train_base_class3 import ModelTrainer
from src.models.baseline import build_baseline, build_compile_baseline
from src.models.model_dropout import build_model_with_dropout, build_compile_dropout
from src.models.model_no_dropout import build_model_no_dropout, build_compile_no_dropout


import argparse


# dsp. va a scripts

def main():
    filepath = 'data/data.csv'
    
    # std data
    preprocessor = BreastCancerPreprocessor(batch_size=64)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)



    # code one by one.......
    """
    # baseline
    trainer = ModelTrainer(model_fn=build_baseline, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, model_name="baseline")
    results = trainer.train()
    # without dropout
    trainer_no_dropout = ModelTrainer(model_fn=build_model_no_dropout, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, 
                                    model_name="no_dropout")
    results_no_dropout = trainer_no_dropout.train()
    print(results_no_dropout['model'])

    # with dropout
    trainer_dropout = ModelTrainer(model_fn=build_model_with_dropout, train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, 
                        model_name="dropout")
    results_dropout = trainer_dropout.train()
    print(results_dropout)
    """



    parser = argparse.ArgumentParser(description="Train models with optional flags.")
    parser.add_argument("--baseline", action="store_true", help="Train baseline model")
    parser.add_argument("--no_dropout", action="store_true", help="Train no_dropout model")
    parser.add_argument("--dropout", action="store_true", help="Train dropout model")

    args = parser.parse_args()

    
    # If no flags passed, train all models
    if not any([args.baseline, args.no_dropout, args.dropout]):
        train_flags = {
            "baseline": True,
            "no_dropout": True,
            "dropout": True,
        }
    else:
        train_flags = {
            "baseline": args.baseline,
            "no_dropout": args.no_dropout,
            "dropout": args.dropout,
        }


    dropout_rate = 0.2

    experiments = []

    for name, model_fn in [
        #("baseline", build_baseline),
        #("no_dropout", build_model_no_dropout),
        #("dropout", build_model_with_dropout),
        ("baseline", build_compile_baseline),
        ("no_dropout", build_compile_no_dropout),
        ("dropout", lambda: build_compile_dropout(dropout_rate=dropout_rate)),  # pass dropout
    ]:
        if not train_flags.get(name, False):
            print(f"Skipping training for model: {name}")
            continue

        print(f"Training model: {name}")


        # pass hyperpmas; batch size came from preprocesing
        hyperparams={'batch_size':64} # dropout rate
        # Add dropout_rate only for 'dropout' model
        if name == "dropout":
            hyperparams["dropout_rate"] = dropout_rate

        trainer = ModelTrainer(
            model_fn=model_fn,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            model_name=name,
            data_variant="testing_purposes",
            hyperparameters=hyperparams
        )
        results = trainer.train()
        experiments.append(results)





    # chosse bets models to make hyperparm tuning accoridng to val_f1
    top_models = sorted(
        experiments,
        key=lambda x: x["val_f1"],
        reverse=True
    )[:2]  # top 2 by val_f1




    #for best in top_models:
      #  print(f"Selected: {best['model_name']} with F1={best['val_f1']:.4f}")
        # You can now feed these into your hyperparameter tuning pipeline





if __name__ == '__main__':
    main()




#python3.10 -m src.training.main --baseline --no_dropout --dropout



"""

python3.10 -m src.training.main.py

# scripts/train_dropout.py
from src.training.train_base import train_model
from src.models.model_dropout import build_model
from src.data.data_norm import get_data

if __name__ == "__main__":
    train_model(build_model, get_data, model_name="dropout_norm")

# scripts/train_no_dropout.py
from src.training.train_base import train_model
from src.models.model_no_dropout import build_model
from src.data.data_std import get_data

if __name__ == "__main__":
    train_model(build_model, get_data, model_name="no_dropout_std")

"""