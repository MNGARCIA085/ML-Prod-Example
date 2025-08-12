import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np


from src.models.model_dropout import build_model_with_dropout
from src.models.compile_utils import compile_model



def test_build_model():
    dropout_rate = 0.25
    model = build_model_with_dropout(dropout_rate)

    assert isinstance(model, Sequential)
    assert len(model.layers) == 5

    dropout_layers = [l for l in model.layers if isinstance(l, Dropout)]
    assert len(dropout_layers) == 2
    for layer in dropout_layers:
        assert abs(layer.rate - dropout_rate) < 1e-6

def test_compile_model():
    model = build_model_with_dropout()
    compile_model(model, learning_rate=0.005)

    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
    assert abs(model.optimizer.learning_rate.numpy() - 0.005) < 1e-6

    assert model.loss == 'binary_crossentropy'




    # Dummy input and labels
    dummy_x = np.random.rand(4, 10).astype('float32')
    dummy_y = np.array([0, 1, 0, 1]).astype('float32')

    # Run a single training step (or evaluation)
    model.train_on_batch(dummy_x, dummy_y)

 
    # dummy input to buyild
    #dummy_input = np.random.rand(1, 10).astype('float32')  # batch=1, features=10
    #_ = model.predict(dummy_input)  # build the model by calling predict
    #print(model.summary())

    metric_names = []
    for m in model.metrics:
        if hasattr(m, 'metrics'):
            metric_names.extend([metric.name for metric in m.metrics])
        else:
            metric_names.append(m.name)

    assert 'accuracy' in metric_names
    assert 'precision' in metric_names
    assert 'recall' in metric_names

def test_model_forward_pass():
    model = build_model_with_dropout()
    compile_model(model)

    x = np.random.rand(4, 10).astype('float32')  # 4 samples, 10 features

    preds = model.predict(x)
    assert preds.shape == (4, 1)



#pytest tests/models/test_model_dropout.py -v --tb=long

#pytest tests

#pytest tests -s -v