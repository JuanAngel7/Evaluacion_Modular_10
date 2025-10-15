from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y escalador
model = joblib.load('modelo_evaluacion_final_M10.pkl')
scaler = joblib.load('scaler.pkl')  # ⬅️ ¡Importante!

@app.route('/')
def home():
    return jsonify({
        "message": "API de predicción del conjunto de datos de cáncer de mama (Kaggle)",
        "instructions": "Envía una petición POST a /predict con el cuerpo: {\"features\": [lista de 30 números]}",
        "example_prediction_endpoint": "GET /predict_example para una predicción con datos de ejemplo."
    })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            "message": "Usa POST para enviar datos de entrada.",
            "example": {"features": [
                17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                1.095, 0.3568, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
                0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
            ]}
        }), 200

    try:
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'error': 'El cuerpo debe contener una clave "features" con una lista de 30 números.'}), 400

        features = np.array(data['features']).reshape(1, -1)
        if features.shape[1] != 30:
            return jsonify({'error': f'Se esperan exactamente 30 características, pero se recibieron {features.shape[1]}.'}), 400

        # 🔑 Aplicar el escalador guardado
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        return jsonify({'prediction': int(prediction[0])})  # Convertir a int para JSON limpio

    except Exception as e:
        return jsonify({'error': 'Error interno del servidor.', 'details': str(e)}), 500

@app.route('/predict_example', methods=['GET'])
def predict_example():
    try:
        example_features = np.array([[
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
            1.095, 0.3568, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003,
            0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]])

        # 🔑 Escalar también en el ejemplo
        example_scaled = scaler.transform(example_features)
        prediction = model.predict(example_scaled)
        return jsonify({'example_prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': 'Error interno del servidor.', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
