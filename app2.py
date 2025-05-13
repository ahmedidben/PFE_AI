from flask import Flask, request, jsonify, render_template
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and label encoder for LSTM
with open('tokenizer.pkl', 'rb') as f:
    tokenizer_lstm = pickle.load(f)

label_encoder = joblib.load('label_encoder.pkl')

# Load vectorizer for ML models
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = joblib.load(f)

app = Flask(__name__)

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Load models
models = {
    'biobert': load_model('best_model2.keras'),        # Expects (768,)
    'lstm': load_model('best_model.keras'),            # Expects (148,)
    'adaboost': joblib.load('adaboost_model.pkl'),
    'random_forest': joblib.load('random_forest_model.pkl'),
    'RNN': load_model('rnn_model.keras'),              # Expects (100,)
    'gradient_boost': joblib.load('gradient_boosting_model.pkl')
}

# Load labels
disease_labels = pd.read_csv('disease_labels.csv', header=None)[0].tolist()

# ----------- Text preprocessing -------------
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # shape: (768,)

def preprocess_for_lstm(text):
    sequence = tokenizer_lstm.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=148)
    return padded  # shape: (1, 148)

# ------------ Flask routes ------------------
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/interface')
def interface():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data or 'text' not in data or 'model' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    input_text = data['text']
    model_name = data['model']

    if model_name not in models:
        return jsonify({'error': 'Invalid model selected'}), 400

    model = models[model_name]

    # Input vector generation
    if model_name == 'biobert':
        input_vector = np.array([embed_text(input_text)])
    elif model_name in ['lstm', 'RNN']:
        input_vector = preprocess_for_lstm(input_text)
    elif model_name in ['adaboost', 'random_forest', 'gradient_boost']:
        input_vector = tfidf_vectorizer.transform([input_text]).toarray()
    else:
        return jsonify({'error': 'Model not supported'}), 400

    # Prediction logic
    try:
        if model_name in ['lstm', 'RNN', 'biobert']:
            prediction = model.predict(input_vector)
            if prediction.ndim == 2:
                predicted_index = int(np.argmax(prediction[0]))
                confidence = float(prediction[0][predicted_index])
            else:
                return jsonify({'error': 'Unexpected prediction shape for deep model'}), 500

        elif model_name in ['adaboost', 'random_forest', 'gradient_boost']:
            prediction_proba = model.predict_proba(input_vector)
            predicted_index = int(np.argmax(prediction_proba[0]))
            confidence = float(prediction_proba[0][predicted_index])

        else:
            return jsonify({'error': 'Unsupported model type'}), 500

        predicted_label = disease_labels[predicted_index]

        return jsonify({
            'prediction': predicted_label,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
