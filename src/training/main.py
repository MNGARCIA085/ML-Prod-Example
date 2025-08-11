from src.data.preprocessor import BreastCancerPreprocessor
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized

from src.training.train_base import train_model
from src.models.model_dropout import build_model_with_dropout
from src.models.model_no_dropout import build_model_no_dropout

def main():
    filepath = 'data/data.csv'
    
    # std data
    preprocessor = BreastCancerPreprocessor(batch_size=64)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

    # normalized data
    #preprocessor = BreastCancerPreprocessorNormalized(batch_size=64)
    #train_ds_norm, val_ds_norm, test_ds_norm = preprocessor.get_datasets(filepath)

   

    # train models
    train_model(build_model_with_dropout, train_ds, val_ds, test_ds, model_name="dropout_norm")


    #
    print('model no dropout')
    train_model(build_model_no_dropout, train_ds, val_ds, test_ds, model_name="no_dropout_norm")









if __name__ == '__main__':
    main()






"""
# scripts/train_dropout.py
from src.training.train_base import train_model
from src.models.model_dropout import build_model
from src.data.data_norm import get_data

if __name__ == "__main__":
    train_model(build_model, get_data, model_name="dropout_norm")

# scripts/train_no_dropout.py
from src.training.train_base import train_model
from src.models.model_no_dropout import build_model
from src.data.data_std import get_data

if __name__ == "__main__":
    train_model(build_model, get_data, model_name="no_dropout_std")

"""