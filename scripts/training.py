
import argparse
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized
from src.training.trainer import ModelTrainer
from src.models.baseline import build_compile_baseline
from src.models.model_dropout import build_compile_dropout
from src.models.model_no_dropout import build_compile_no_dropout
from src.common.callbacks import get_callbacks
from src.config.constants import BREAST_CANCER_CSV_RAW
from src.training.experiment_runner import run_experiment
from src.models.factory import get_model_fns
from src.data.utils import get_preprocessors





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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs (default: 2)")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    

    args = parser.parse_args()

    # Defaults if not provided
    models_to_train = args.models or ["baseline", "no_dropout", "dropout"]
    data_variants = args.data_variants or ["simple", "standardize"]

    # preprocessors
    preprocessors = get_preprocessors(data_variants, args.batch_size)
    
    # Model function mapping
    model_fns = get_model_fns(args.dropout_rate)

    # callbacks
    callbacks = get_callbacks(
        use_early_stopping=True, 
        use_reduce_lr=False
    )


    # run experiments
    experiments = run_experiment(model_fns,
                             preprocessors, filepath,
                             epochs=args.epochs, callbacks=callbacks,
                             batch_size=args.batch_size,
                             dropout_rate=args.dropout_rate)

    
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