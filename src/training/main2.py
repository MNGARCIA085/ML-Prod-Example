from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized

#from src.training.train_base import train_model


from src.training.train_base_class3 import ModelTrainer
from src.models.baseline import build_baseline, build_compile_baseline
from src.models.model_dropout import build_model_with_dropout, build_compile_dropout
from src.models.model_no_dropout import build_model_no_dropout, build_compile_no_dropout


import argparse



def main():
    filepath = 'data/data.csv'
    
    # All preprocessing variants you want to test
    preprocessors = [
        ("simple", BreastCancerPreprocessor(batch_size=64)),
        ("normalized", BreastCancerPreprocessorNormalized(batch_size=64))
    ]

    parser = argparse.ArgumentParser(description="Train models with optional flags.")
    parser.add_argument("--baseline", action="store_true", help="Train baseline model")
    parser.add_argument("--no_dropout", action="store_true", help="Train no_dropout model")
    parser.add_argument("--dropout", action="store_true", help="Train dropout model")
    args = parser.parse_args()

    # If no flags passed, train all models
    if not any([args.baseline, args.no_dropout, args.dropout]):
        train_flags = {"baseline": True, "no_dropout": True, "dropout": True}
    else:
        train_flags = {
            "baseline": args.baseline,
            "no_dropout": args.no_dropout,
            "dropout": args.dropout,
        }

    dropout_rate = 0.2
    experiments = []

    # Loop over preprocessing variants
    for data_variant, preprocessor in preprocessors:
        train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

        for name, model_fn in [
            ("baseline", build_compile_baseline),
            ("no_dropout", build_compile_no_dropout),
            ("dropout", lambda: build_compile_dropout(dropout_rate=dropout_rate)),
        ]:
            if not train_flags.get(name, False):
                print(f"Skipping training for model: {name}")
                continue

            print(f"Training model: {name} | data: {data_variant}")

            hyperparams = {'batch_size': 64}
            if name == "dropout":
                hyperparams["dropout_rate"] = dropout_rate

            trainer = ModelTrainer(
                model_fn=model_fn,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                model_name=name,
                data_variant=data_variant,  # store preprocessing type
                hyperparameters=hyperparams
            )
            results = trainer.train()
            experiments.append(results)

    # Select best models by validation F1
    top_models = sorted(experiments, key=lambda x: x["val_f1"], reverse=True)[:2]


if __name__ == '__main__':
    main()
