from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
CORS(app)

# Load TFLite model
interpreter = tf.lite.Interpreter(
    model_path="phq9_cnn_with_demographics(2).tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [
    "Minimal Depression",
    "Mild Depression",
    "Moderate Depression",
    "Moderately Severe Depression",
    "Severe Depression"
]

@app.route("/")
def home():
    return "CNN PHQ-9 API Running"

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        features = np.array(
            data["features"],
            dtype=np.float32
        )

        # 11 input features
        features = features.reshape(1, 11, 1)

        interpreter.set_tensor(
            input_details[0]['index'],
            features
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]['index']
        )

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": labels[predicted_class],
            "confidence": confidence,
            "probabilities": prediction.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
