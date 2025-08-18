from src.data.base_preprocessor import BasePreprocessor

class BreastCancerPreprocessor(BasePreprocessor):
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
