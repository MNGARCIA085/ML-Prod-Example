from src.data.base_preprocessor import BasePreprocessor

class BreastCancerPreprocessor(BasePreprocessor):


    def __init__(self, batch_size=32, val_size=0.1, random_state=42):
        super().__init__(batch_size, val_size, random_state)
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


