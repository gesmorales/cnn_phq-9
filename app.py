from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(
    model_path="phq9_cnn_with_demographics.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/")
def home():
    return {
        "message": "CNN PHQ-9 API is running"
    }


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Example:
        # {
        #   "features": [1,2,3,4,5,...]
        # }

        features = np.array(
            data["features"],
            dtype=np.float32
        )

        # reshape for CNN input
        input_data = np.expand_dims(features, axis=0)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        output_data = interpreter.get_tensor(
            output_details[0]["index"]
        )

        prediction = output_data.tolist()

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
