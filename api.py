from flask import Flask, request, jsonify
from model_loader import load_model
import numpy as np

app = Flask(__name__)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        data['Time'],
        data['V1'], data['V2'], data['V3'], data['V4'], data['V5'],
        data['V6'], data['V7'], data['V8'], data['V9'], data['V10'],
        data['V11'], data['V12'], data['V13'], data['V14'], data['V15'],
        data['V16'], data['V17'], data['V18'], data['V19'], data['V20'],
        data['V21'], data['V22'], data['V23'], data['V24'], data['V25'],
        data['V26'], data['V27'], data['V28'], data['Amount']
    ]])
    
    prediction = model.predict_proba(features)[0][1]
    return jsonify({'fraud_probability': float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
