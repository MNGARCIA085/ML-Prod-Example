from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from src.models.utils import compile_model


# build model
def build_model_no_dropout():
    model = Sequential([
        #Input(shape=(None,)),  # None means "any number of features"
        Dense(16, activation='relu'),
        #Dropout(0.1),
        #Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name='no_dropout')
    return model


# build and compile
def build_compile_no_dropout():
    model = build_model_no_dropout()
    compile_model(model)
    return model


# for keras tuner
def build_model_no_dropout_tuner(hp): #, input_dim):
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    model = build_model_no_dropout()
    compile_model(model, learning_rate)
    return model