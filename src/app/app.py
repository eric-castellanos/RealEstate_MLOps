from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("././models/housing_model.keras", compile=False)

# Recompile with the current optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="mse",
              metrics=["mae", "mse"])

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello World!"


@app.route("/serve_model", methods=["GET"])
def serve():
    try:
        # Get query parameters
        features = request.args.getlist("feature", type=float)
        
        if len(features) != 13:  # Boston Housing dataset has 13 features
            return jsonify({"error": "Expected 13 feature values"}), 400
        
        # Convert input to a NumPy array and reshape for prediction
        input_data = np.array([features])
        
        # Make prediction
        prediction = model.predict(input_data)[0][0]

        return jsonify({"prediction": float(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/model_metadata", methods=["GET"])
def get_model_metadata():
    pass

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
