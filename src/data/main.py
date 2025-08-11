from src.data.preprocessor import BreastCancerPreprocessor
# or
# from src.data.breast_cancer_preprocessor_with_normalization import BreastCancerPreprocessorNormalized

def main():
    filepath = 'data/data.csv'
    preprocessor = BreastCancerPreprocessor(batch_size=64)
    train_ds, val_ds, test_ds = preprocessor.get_datasets(filepath)

    # Now use train_ds, val_ds, test_ds for your model training/testing
    for batch in train_ds.take(1):
        X_batch, y_batch = batch
        print(X_batch, y_batch)
        print(X_batch.shape, y_batch.shape)

    print('sanity check')

if __name__ == '__main__':
    main()
