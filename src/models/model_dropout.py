from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from src.models.utils import compile_model


# build model
def build_model_with_dropout(dropout_rate=0.1):
    model = Sequential([
        #Input(shape=(None,)),  # None means "any number of features"
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ],name='dropout')
    return model


# build and compile
def build_compile_dropout(dropout_rate=0.1):
    model = build_model_with_dropout()
    compile_model(model)
    return model


# for keras tuner
def build_model_with_dropout_tuner(hp): #, input_dim):
    dropout_rate = hp.Float("dropout_rate", 0.0, 0.5, step=0.1)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    model = build_model_with_dropout(dropout_rate=dropout_rate)

    # Recompile model with tunable learning rate
    compile_model(model, learning_rate)
    return model




