import tensorflow as tf

def compile_model(model, learning_rate=1e-1): # not better lr to start to show more training later
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )