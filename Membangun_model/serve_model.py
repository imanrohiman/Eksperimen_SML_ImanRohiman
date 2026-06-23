import mlflow.sklearn
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import json
import os

app = Flask(__name__)

# Load model dari MLflow (pakai Run ID yang sudah ada)
RUN_ID = "66ddba9472a14482bd44e58f37b4045b"
model_path = f"runs:/{RUN_ID}/model"

print(f"📂 Loading model from: {model_path}")
model = mlflow.sklearn.load_model(model_path)
print("✅ Model loaded successfully!")

@app.route('/invocations', methods=['POST'])
def predict():
    try:
        # Parse request
        data = request.get_json()
        
        # Handle different input formats
        if 'dataframe_split' in data:
            # MLflow format
            df = pd.DataFrame(
                data['dataframe_split']['data'],
                columns=data['dataframe_split']['columns']
            )
        elif 'instances' in data:
            # TensorFlow format
            df = pd.DataFrame(data['instances'])
        else:
            # Simple format
            df = pd.DataFrame(data)
        
        # Predict
        predictions = model.predict(df)
        
        # Return result
        return jsonify({
            'predictions': predictions.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'Titanic Survival Model',
        'run_id': RUN_ID
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': '🚀 Titanic Survival Model Serving',
        'run_id': RUN_ID,
        'accuracy': '78.77%',
        'endpoints': {
            '/invocations': 'POST - Send data for prediction',
            '/health': 'GET - Check service health'
        },
        'example': {
            'url': 'http://127.0.0.1:5001/invocations',
            'method': 'POST',
            'body': {
                'dataframe_split': {
                    'columns': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
                    'data': [[3, 1, -0.5, 1, 0, -0.5, 2]]
                }
            }
        }
    })

if __name__ == '__main__':
    print("="*50)
    print("🚀 TITANIC MODEL SERVING")
    print("="*50)
    print(f"📍 Run ID: {RUN_ID}")
    print(f"🔗 Serving on: http://127.0.0.1:5001")
    print(f"📊 Endpoint: http://127.0.0.1:5001/invocations")
    print(f"🔍 Health: http://127.0.0.1:5001/health")
    print("="*50)
    print("Press Ctrl+C to stop")
    print("="*50)
    app.run(host='0.0.0.0', port=5001, debug=False)
