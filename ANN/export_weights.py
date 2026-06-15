import tensorflow as tf
import joblib

model = tf.keras.models.load_model(
    "models/breast_cancer_ann.keras",
    compile=False
)

weights = model.get_weights()

joblib.dump(weights, "models/weights.pkl")

print("Weights saved!")