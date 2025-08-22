import os
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from src.config.constants import BREAST_CANCER_CSV_RAW_TEST


# load data
def load_data(scaler_path: str, encoder_path: str):
    # read CSV
    df = pd.read_csv(BREAST_CANCER_CSV_RAW_TEST)
    df = df.drop(columns=["Unnamed: 32"])
    df = df.dropna()

    # labels
    encoder = joblib.load(encoder_path)
    y_test = df["diagnosis"].map(encoder)

    # features
    features = df.drop(columns=["id", "diagnosis"])

    # scale features
    scaler = joblib.load(scaler_path)
    X_test = pd.DataFrame(
        scaler.transform(features),
        columns=features.columns,
        index=features.index,
    )

    return X_test, y_test



# evaluate
def evaluate(experiment_path: str):
    print(f"Evaluating experiment: {experiment_path}")

    # load model
    model_path = os.path.join(experiment_path, "model.h5")
    model = keras.models.load_model(model_path)

    # load test data
    scaler_path = os.path.join(experiment_path, "scaler.pkl")
    encoder_path = os.path.join(experiment_path, "encoder.pkl")
    X_test, y_test = load_data(scaler_path, encoder_path)

    # build dataset for evaluation
    ds = tf.data.Dataset.from_tensor_slices(
        (X_test.values.astype("float32"), y_test.values.astype("int32"))
    ).batch(32)

    # metrics
    res = model.evaluate(ds, verbose=0)
    print("Evaluation metrics:", res)

    # confusion matrix
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    conf_matrix = tf.math.confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", conf_matrix.numpy())

    return experiment_path, res, conf_matrix.numpy()

    # save to file
    #save_metrics(experiment_path, res, conf_matrix.numpy())

