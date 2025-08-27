from .baseline import build_compile_baseline
from .model_dropout import build_compile_dropout
from .model_no_dropout import build_compile_no_dropout


# Model function mapping
def get_model_fns(dropout_rate=0.2):
    return {
        "baseline": build_compile_baseline,
        "no_dropout": build_compile_no_dropout,
        "dropout": lambda: build_compile_dropout(dropout_rate=dropout_rate)
    }
