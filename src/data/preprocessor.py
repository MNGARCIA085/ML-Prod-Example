from src.data.base_preprocessor import BasePreprocessor

class BreastCancerPreprocessor(BasePreprocessor):


    def __init__(self, batch_size=32, test_size=0.2, val_size=0.1, random_state=42):
        super().__init__(batch_size, test_size, val_size, random_state)
        self.scaler = None
        self.encoder = None 
        self.feature_columns = None

    def preprocess_data(self):
        """
        df = self.df.copy()
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
        df = df.dropna()
        labels = df['diagnosis'].map({'M': 1, 'B': 0})
        features = df.drop(columns=['id', 'diagnosis'])
        return features, labels
        """
        df = self.df.copy()
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
        df = df.dropna()
        
        # Define mapping and store it
        self.encoder = {'M': 1, 'B': 0}
        labels = df['diagnosis'].map(self.encoder)

        features = df.drop(columns=['id', 'diagnosis'])
        return features, labels


#https://chatgpt.com/c/68a65eea-2794-8324-b403-873c094f7d7c