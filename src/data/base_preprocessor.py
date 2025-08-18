import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class BasePreprocessor:
    """
    A generic preprocessing pipeline for tabular datasets.

    Responsibilities:
    - Load raw data from a CSV file.
    - Provide hooks (`preprocess_data`, `post_split_transform`) for dataset-specific
      preprocessing steps.
    - Split data into training, validation, and test sets with stratification.
    - Convert data into TensorFlow `tf.data.Dataset` objects with batching and shuffling.

    Intended Usage:
    - Inherit from this class and override:
        * `preprocess_data()` to define how to extract features and labels.
        * `post_split_transform()` if additional transformations are needed
          after splitting (e.g., normalization, scaling).
    """

    
    def __init__(self, batch_size=32, test_size=0.2, val_size=0.1, random_state=42):
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def load_data(self, filepath):
        self.df = pd.read_csv(filepath)

    def preprocess_data(self):
        """
        Override this in subclasses.
        Should return features and labels.
        """
        raise NotImplementedError

    def split_data(self, features, labels):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state
        )
        val_relative_size = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_relative_size,
            stratify=y_train_val,
            random_state=self.random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def post_split_transform(self, X_train, X_val, X_test):
        """Override this to transform after splitting, e.g. scaling."""
        return X_train, X_val, X_test

    def tf_dataset(self, X, y, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((X.values.astype('float32'), y.values.astype('int32')))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X))
        return ds.batch(self.batch_size)

    def get_datasets(self, filepath):
        self.load_data(filepath)
        features, labels = self.preprocess_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(features, labels)
        X_train, X_val, X_test = self.post_split_transform(X_train, X_val, X_test)
        train_ds = self.tf_dataset(X_train, y_train, shuffle=True)
        val_ds = self.tf_dataset(X_val, y_val, shuffle=False)
        test_ds = self.tf_dataset(X_test, y_test, shuffle=False)
        return train_ds, val_ds, test_ds
