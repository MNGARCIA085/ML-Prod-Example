from src.data.base_preprocessor import BasePreprocessor

class BreastCancerPreprocessor(BasePreprocessor):

    def preprocess_data(self):
        df = self.df.copy()
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
        df = df.dropna()
        labels = df['diagnosis'].map({'M': 1, 'B': 0})
        features = df.drop(columns=['id', 'diagnosis'])
        return features, labels
