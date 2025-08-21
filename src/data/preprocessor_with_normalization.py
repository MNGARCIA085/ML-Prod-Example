from sklearn.preprocessing import StandardScaler
from src.data.base_preprocessor import BasePreprocessor
import pandas as pd




class BreastCancerPreprocessorNormalized(BasePreprocessor):
    """
    Preprocessor for the Breast Cancer dataset with normalization.

    - Cleans data, encodes labels.
    - Standardizes features (zero mean, unit variance).
    - Saves fitted scaler + feature column order for inference consistency.
    """

    def __init__(self, batch_size=32, test_size=0.2, val_size=0.1, random_state=42):
        super().__init__(batch_size, test_size, val_size, random_state)
        self.scaler = None
        self.encoder = None 
        self.feature_columns = None
        


    def preprocess_data(self):
        df = self.df.copy()
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
        df = df.dropna()
        
        # Define mapping and store it
        self.encoder = {'M': 1, 'B': 0}
        labels = df['diagnosis'].map(self.encoder)

        features = df.drop(columns=['id', 'diagnosis'])
        return features, labels
    

    def post_split_transform(self, X_train, X_val, X_test):
        # Save column order for inference
        self.feature_columns = list(X_train.columns)

        # save the scaler
        self.scaler = StandardScaler()

        # Fit scaler on training set
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        # Apply the scaler to val and test
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_scaled, X_val_scaled, X_test_scaled
