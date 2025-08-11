from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input




def build_model_with_dropout(dropout_rate=0.1):
    model = Sequential([
        #Input(shape=(None,)),  # None means "any number of features"
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(16, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    return model



"""
In your tuning pipeline, pass the tuner wrapper function; in normal training pipeline, use plain model builder.
"""


#


from src.models.compile_utils import compile_model



def build_model_with_dropout_tuner(hp): #, input_dim):
    dropout_rate = hp.Float("dropout_rate", 0.0, 0.5, step=0.1)
    learning_rate = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")

    model = build_model_with_dropout(dropout_rate=dropout_rate)

    # Recompile model with tunable learning rate
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy','precision','recall']
    )
    """
    compile_model(model, learning_rate)
    return model




"""
input_dim = X_train.shape[1]  # number of features
model = build_model_with_dropout(input_dim)
# or
model = build_model_no_dropout(input_dim)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
"""