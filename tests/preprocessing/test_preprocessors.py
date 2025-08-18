import io
import pytest
import tensorflow as tf
import pandas as pd
from src.data.base_preprocessor import BasePreprocessor
from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized


# -----------------------------
# Minimal subclass for testing
# -----------------------------
class TestPreprocessor(BasePreprocessor):
    def preprocess_data(self):
        X = self.df.drop(columns=['id', 'diagnosis'])
        y = self.df['diagnosis'].map({'B': 0, 'M': 1})
        return X, y


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_csv():
    data = """id,feature1,feature2,diagnosis
1,10,0.5,B
2,15,0.7,M
3,10,0.8,B
4,13,0.2,M
5,14,0.4,B
6,12,0.3,M
7,16,0.9,B
8,11,0.1,M
9,13,0.5,B
10,10,0.6,M
11,14,0.7,B
12,13,0.8,M
"""
    return io.StringIO(data)


# -----------------------------
# Base preprocessor tests
# -----------------------------
def test_get_datasets(sample_csv):
    preprocessor = TestPreprocessor(batch_size=2, test_size=0.2, val_size=0.25, random_state=0)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(sample_csv)

    # Check types
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    # Check batch shapes and dtypes
    for batch in train_ds.take(1):
        features, labels = batch
        assert features.shape[1] == 2  # 2 features
        assert features.dtype == tf.float32
        assert labels.dtype == tf.int32


def test_splits_sum_correctly(sample_csv):
    preprocessor = TestPreprocessor(batch_size=2, test_size=0.2, val_size=0.25, random_state=0)
    preprocessor.load_data(sample_csv)  # <-- add this line
    features, labels = preprocessor.preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(features, labels)

    total_rows = len(features)
    assert len(X_train) + len(X_val) + len(X_test) == total_rows
    assert len(y_train) + len(y_val) + len(y_test) == total_rows



# -----------------------------
# Normalized preprocessor tests
# -----------------------------
def test_normalization_applied(sample_csv):  # use the same fixture
    preprocessor = BreastCancerPreprocessorNormalized(batch_size=4)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(sample_csv)

    x_batch, _ = next(iter(train_ds))
    mean_val = float(x_batch.numpy().mean())
    assert abs(mean_val) < 1.0
    assert x_batch.dtype == tf.float32



#pytest tests/preprocessing
#pytest -q tests/prepreocessing/test_preprocessors.py -v
