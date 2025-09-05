
import argparse
from src.common.callbacks import get_callbacks
from src.config.constants import BREAST_CANCER_CSV_RAW
from src.training.experiment_runner import run_experiment
from src.models.factory import get_model_fns
from src.data.utils import get_preprocessors
from src.training.utils import get_top_n_models



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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs (default: 10)")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    

    args = parser.parse_args()

    # Defaults if not provided
    data_variants = args.data_variants or ["simple", "standardize"]

    # preprocessors
    preprocessors = get_preprocessors(data_variants, args.batch_size)
    
    # Model function mapping
    model_fns = get_model_fns(args.dropout_rate)

    # callbacks
    callbacks = get_callbacks(
        use_early_stopping=False, 
        use_reduce_lr=False
    )


    # run experiments
    results = run_experiment(model_fns,
                             preprocessors, filepath,
                             epochs=args.epochs, callbacks=callbacks,
                             batch_size=args.batch_size,
                             dropout_rate=args.dropout_rate)


    # select best experiments
    top_models = get_top_n_models(results, recall_threshold=0.8, top_n=2)

    # print top models
    for m in top_models:
        print(f"{m['model_name']} ({m['data_variant']}): recall={m['val_recall']:.3f}, f1={m['val_f1']:.3f}")




if __name__ == '__main__':
    main()


"""
python -m scripts.training --epochs 20 --batch_size 64

python train.py --models baseline dropout --data_variants simple standardize
python train.py  # trains all models with all data variants
python train.py --batch_size 128 --dropout_rate 0.3

"""