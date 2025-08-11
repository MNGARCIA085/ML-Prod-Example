from sklearn.preprocessing import StandardScaler
from src.data.base_preprocessor import BasePreprocessor

class BreastCancerPreprocessorNormalized(BasePreprocessor):
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
