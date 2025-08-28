from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.models.utils import compile_model


# simple baseline model
def build_baseline():
    model = Sequential([
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ],name='baseline')
    return model


# build and compile
def build_compile_baseline(dropout_rate=0.1):
    model = build_baseline()
    compile_model(model)
    return model


# for Keras tuner
def baseline_with_tuner(hp):
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    model = build_baseline()

    # Recompile model with tunable learning rate
    compile_model(model, learning_rate)
    return model



