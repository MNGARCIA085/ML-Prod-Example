from .baseline import build_compile_baseline, baseline_with_tuner
from .model_dropout import build_compile_dropout, build_model_with_dropout_tuner
from .model_no_dropout import build_compile_no_dropout, build_model_no_dropout_tuner


# Model function mapping
def get_model_fns(dropout_rate=0.2):
    return {
        "baseline": build_compile_baseline,
        "no_dropout": build_compile_no_dropout,
        "dropout": lambda: build_compile_dropout(dropout_rate=dropout_rate)
    }


# Model function mapping for the tuner
def get_model_fns_tuner():
    return {
        "baseline": baseline_with_tuner,
        "no_dropout": build_model_with_dropout_tuner,
        "dropout": build_model_no_dropout_tuner
    }
