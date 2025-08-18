from sklearn.preprocessing import StandardScaler
from src.data.base_preprocessor import BasePreprocessor
import pandas as pd

class BreastCancerPreprocessorNormalized(BasePreprocessor):
    """
    Preprocessor for the Breast Cancer dataset with normalization.

    Inherits the same data cleaning and label encoding steps as
    `BreastCancerPreprocessor`, but additionally overrides
    `post_split_transform` to apply feature scaling.

    Scaling:
    - Standardizes features (zero mean, unit variance) using
      `sklearn.preprocessing.StandardScaler`.
    - Fits the scaler on the training set, then applies it to
      validation and test sets.
    """

    
    def preprocess_data(self):
        df = self.df.copy()
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
        df = df.dropna()
        labels = df['diagnosis'].map({'M': 1, 'B': 0})
        features = df.drop(columns=['id', 'diagnosis'])
        return features, labels

    def post_split_transform(self, X_train, X_val, X_test):
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_val_scaled, X_test_scaled
