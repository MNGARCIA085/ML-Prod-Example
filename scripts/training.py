
import argparse
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.training.trainer import ModelTrainer
from src.models.baseline import build_compile_baseline
from src.models.model_dropout import build_compile_dropout
from src.models.model_no_dropout import build_compile_no_dropout
from src.common.callbacks import get_callbacks
from src.config.constants import BREAST_CANCER_CSV_RAW




def get_preprocessor(name, batch_size):
    mapping = {
        "simple": BreastCancerPreprocessor(batch_size=batch_size),
        "standardize": BreastCancerPreprocessorNormalized(batch_size=batch_size),
    }
    if name not in mapping:
        raise ValueError(f"Unknown data variant: {name}")
    return mapping[name]


def main():
    filepath = BREAST_CANCER_CSV_RAW
    

    # CLI setup
    parser = argparse.ArgumentParser(description="Train models with preprocessing variants.")
    # models to train
    parser.add_argument("--models", nargs="+", choices=["baseline", "no_dropout", "dropout"],
                        help="Models to train (default: all)")
    # data variants
    parser.add_argument("--data_variants", nargs="+", choices=["simple", "standardize"],
                        help="Data variants to use (default: all)")

    # hyperparams
    parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for training (default: 64)")
    parser.add_argument("--dropout_rate", type=float, default=0.2,
                    help="Dropout rate (default: 0.2)")
    

    args = parser.parse_args()

    # Defaults if not provided
    models_to_train = args.models or ["baseline", "no_dropout", "dropout"]
    data_variants = args.data_variants or ["simple", "standardize"]



    # callbacks
    callbacks = get_callbacks(
        use_early_stopping=True, 
        use_reduce_lr=False
    )

    
    # Model function mapping
    model_fns = {
        "baseline": build_compile_baseline,
        "no_dropout": build_compile_no_dropout,
        "dropout": lambda: build_compile_dropout(dropout_rate=args.dropout_rate)
    }

    experiments = []


    for data_variant in data_variants:
        preprocessor = get_preprocessor(data_variant, args.batch_size)
        train_ds, val_ds = preprocessor.get_datasets(filepath)

        for model_name in models_to_train:
            print(f"Training model: {model_name} | data: {data_variant}")

            hyperparams = {"batch_size": args.batch_size}
            if model_name == "dropout":
                hyperparams["dropout_rate"] = args.dropout_rate

            trainer = ModelTrainer(
                model_fn=model_fns[model_name],
                train_ds=train_ds,
                val_ds=val_ds,
                model_name=model_name,
                data_variant=data_variant,
                epochs=2,
                hyperparameters=hyperparams,
                callbacks=callbacks
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
python train.py --models baseline dropout --data_variants simple standardize
python train.py  # trains all models with all data variants
python train.py --batch_size 128 --dropout_rate 0.3


python -m scripts.training

"""