import io
import pandas as pd
import pytest
import tensorflow as tf
from src.data.base_preprocessor import BasePreprocessor  # Adjust import accordingly

# Minimal subclass for testing
class TestPreprocessor(BasePreprocessor):
    def preprocess_data(self):
        # Use dataframe loaded in self.df, separate features and labels
        X = self.df.drop(columns=['id', 'label'])
        y = self.df['label'].map({'B': 0, 'M': 1})  # simple encoding
        return X, y

@pytest.fixture
def sample_csv():
    data = """id,feature1,feature2,label
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
"""
    return io.StringIO(data)

def test_get_datasets(sample_csv):
    preprocessor = TestPreprocessor(batch_size=2, test_size=0.2, val_size=0.25, random_state=0)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(sample_csv)

    # Check types
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    # Check batch shapes and content
    for batch in train_ds.take(1):
        features, labels = batch
        assert features.shape[1] == 2  # 2 features
        assert features.dtype == tf.float32
        assert labels.dtype == tf.int32


# pytest tests/preprocessing