
import argparse
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized

from src.training.train_base_class3 import ModelTrainer
from src.models.baseline import build_compile_baseline
from src.models.model_dropout import build_compile_dropout
from src.models.model_no_dropout import build_compile_no_dropout


def get_preprocessor(name, batch_size):
    mapping = {
        "simple": BreastCancerPreprocessor(batch_size=batch_size),
        "normalized": BreastCancerPreprocessorNormalized(batch_size=batch_size),
    }
    if name not in mapping:
        raise ValueError(f"Unknown data variant: {name}")
    return mapping[name]


def main():
    filepath = 'data/data.csv'
    batch_size = 64
    dropout_rate = 0.2

    # CLI setup
    parser = argparse.ArgumentParser(description="Train models with preprocessing variants.")
    parser.add_argument("--models", nargs="+", choices=["baseline", "no_dropout", "dropout"],
                        help="Models to train (default: all)")
    parser.add_argument("--data_variants", nargs="+", choices=["simple", "normalized"],
                        help="Data variants to use (default: all)")
    args = parser.parse_args()

    # Defaults if not provided
    models_to_train = args.models or ["baseline", "no_dropout", "dropout"]
    data_variants = args.data_variants or ["simple", "normalized"]

    # Model function mapping
    model_fns = {
        "baseline": build_compile_baseline,
        "no_dropout": build_compile_no_dropout,
        "dropout": lambda: build_compile_dropout(dropout_rate=dropout_rate)
    }

    experiments = []

    for data_variant in data_variants:
        preprocessor = get_preprocessor(data_variant, batch_size)
        train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

        for model_name in models_to_train:
            print(f"Training model: {model_name} | data: {data_variant}")

            hyperparams = {"batch_size": batch_size}
            if model_name == "dropout":
                hyperparams["dropout_rate"] = dropout_rate

            trainer = ModelTrainer(
                model_fn=model_fns[model_name],
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                model_name=model_name,
                data_variant=data_variant,
                hyperparameters=hyperparams
            )
            results = trainer.train()
            experiments.append(results)

    # Pick top 2 by validation F1
    top_models = sorted(experiments, key=lambda x: x["val_f1"], reverse=True)[:2]

    print("\nTop models:")
    for best in top_models:
        print(f"{best['model_name']} ({best['data_variant']}): F1={best['val_f1']:.4f}")


if __name__ == '__main__':
    main()











"""
python train.py --models baseline dropout --data_variants simple normalized


python train.py  # trains all models with all data variants
"""