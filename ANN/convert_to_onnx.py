import tensorflow as tf
import tf2onnx

# Load model
model = tf.keras.models.load_model("models/breast_cancer_ann.keras")

# Define input signature
spec = (tf.TensorSpec((None, model.input_shape[1]), tf.float32, name="input"),)

# Convert
model_proto, _ = tf2onnx.convert.from_function(
    model,
    input_signature=spec,
    output_path="models/breast_cancer_ann.onnx"
)

print("ONNX model saved successfully!")