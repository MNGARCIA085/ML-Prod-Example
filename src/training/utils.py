import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List, Dict



# optimizer info
def extract_optimizer_info(model):
    opt = model.optimizer
    if hasattr(opt, "_name"):
        opt_name = opt._name
    elif hasattr(opt, "get_config"):
        config = opt.get_config()
        opt_name = config.get("name", "unknown")
    else:
        opt_name = opt.__class__.__name__
    try:
        lr = float(tf.keras.backend.get_value(opt.learning_rate))
    except Exception:
        lr = None
    return {"optimizer": opt_name, "learning_rate": lr}



# get best models according to our criterion
def get_top_n_models(results: List[Dict], recall_threshold: float = 0.2, top_n: int = 2):
    """
    Select top N models according to recall + F1.

    Args:
        results: List of result dicts (like your example).
        recall_threshold: Minimum recall required.
        top_n: Number of top models to return.

    Returns:
        List of top N result dicts sorted by F1 descending.
    """
    # Filter models that meet the recall threshold
    eligible = [r for r in results if r["val_recall"] >= recall_threshold]

    # Sort by F1 descending
    eligible_sorted = sorted(eligible, key=lambda r: r["val_f1"], reverse=True)

    # Return top N
    return eligible_sorted[:top_n]
