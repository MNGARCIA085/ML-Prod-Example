from src.data.preprocessor import BreastCancerPreprocessor
# or
from src.data.preprocessor_with_normalization import BreastCancerPreprocessorNormalized

def main():
    filepath = 'data/data.csv'

    
    # simple preprocessor
    preprocessor = BreastCancerPreprocessor(batch_size=64)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)
    
    # Now use train_ds, val_ds, test_ds for your model training/testing
    for batch in train_ds.take(1):
        X_batch, y_batch = batch
        print('--Simple preprocessor--')
        #print(X_batch, y_batch)
        print(X_batch.shape, y_batch.shape)



    # simple preprocessor
    preprocessor = BreastCancerPreprocessorNormalized(batch_size=32)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

    # Now use train_ds, val_ds, test_ds for your model training/testing
    for batch in train_ds.take(1):
        X_batch, y_batch = batch
        print('--Preprocessor with normalization--')
        #print(X_batch, y_batch)
        print(X_batch.shape, y_batch.shape)

if __name__ == '__main__':
    main()
