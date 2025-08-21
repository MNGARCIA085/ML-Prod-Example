
import os
import joblib
import tensorflow as tf
from tensorflow import keras

import pandas as pd


from src.config.constants import BREAST_CANCER_CSV_RAW_TEST


"""
# load the scaler
scaler = joblib.load("scaler.pkl")

# use it to transform new data
X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns, index=X_new.index)
"""



"""
import joblib
joblib.dump(self.label_mapping, "artifacts/label_mapping.pkl")
# Later
self.label_mapping = joblib.load("artifacts/label_mapping.pkl")
"""


base_path='/home/marcos/Escritorio/AI-prod/ML-Prod-Example/saved_models/experiment_baseline_20250821_191647/'



# evaluate model on test set
def main():
	# load model
	model_path = os.path.join(base_path, 'model.h5')
	model = keras.models.load_model(model_path)


	# preprocess data (drop fature column, standarize accordingly, creat TF dataset)


	df = pd.read_csv(BREAST_CANCER_CSV_RAW_TEST)
	df = df.drop(columns=['Unnamed: 32'])
	df = df.dropna()


	# dsp. con el ecnoder guardado (que acá de hecho es un dict)
	# Define mapping and store it
	encoder = {'M': 1, 'B': 0}
	labels = df['diagnosis'].map(encoder)
	y_test = labels



    # scale fatures
	features = df.drop(columns=['id', 'diagnosis'])
	scaler_path = os.path.join(base_path, 'scaler.pkl')
	scaler = joblib.load(scaler_path)

	X_test = pd.DataFrame(
			scaler.transform(features),
            columns=features.columns,
            index=features.index
	)



	ds = tf.data.Dataset.from_tensor_slices((X_test.values.astype('float32'), y_test.values.astype('int32'))).batch(32)

	


	# compute metrics (model.evaluate)
	res = model.evaluate(ds)
	print(res)



	# for conf. matrix
	y_pred_prob = model.predict(X_test)
	y_pred = (y_pred_prob > 0.5).astype(int)

	

	conf_matrix = tf.math.confusion_matrix(y_test, y_pred)
	print(conf_matrix)

	"""
	from sklearn.metrics import confusion_matrix

	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	"""







if __name__ == "__main__":
	main()




"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 1. Get predicted probabilities for the positive class
y_pred_prob = model.predict(X_test).ravel()  # shape (num_samples,)

# 2. Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# 3. Compute area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# 4. Plot
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


Key points:

y_pred_prob must be probabilities, not thresholded 0/1 predictions.

You can compute AUC with auc(fpr, tpr) or roc_auc_score(y_test, y_pred_prob).

The ROC curve visualizes how well the model separates classes over all thresholds.
"""