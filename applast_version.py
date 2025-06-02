from flask import Flask, request, jsonify, render_template, redirect, url_for
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
import json
import traceback

# Load symptom synonym map
with open('try.json', 'r', encoding='utf-8') as f:
    symptom_map = json.load(f)

# Initialize Flask
app = Flask(__name__)

# -------------------- Load Resources --------------------

tokenizer_bert = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

with open('tokenizer.pkl', 'rb') as f:
    tokenizer_seq = pickle.load(f)

with open('tfidf_vectorizer2.pkl', 'rb') as f:
    tfidf_vectorizer = joblib.load(f)

label_encoder = joblib.load('label_encoder2.pkl')
disease_labels = pd.read_csv('disease_labels.csv', header=None)[0].tolist()

mlp_model_loaded = joblib.load('mlp_models.pkl')
mlp_model = mlp_model_loaded['model'] if isinstance(mlp_model_loaded, dict) else mlp_model_loaded

models = {
    'biobert': load_model('best_model2.keras'),
    'RNN': load_model('rnn_model.keras'),
    'TextCNN': load_model('textcnn_medical_symptoms.h5'),
    'BiLSTM': load_model('bilstm_medical_symptoms.h5'),
    'mlp': mlp_model,
    'adaboost': joblib.load('adaboost_model.pkl'),
    'random_forest': joblib.load('random_forest_model_2a.pkl'),
    'gradient_boost': joblib.load('gradient_boosting_model2.pkl')
}

# ------------------- Preprocessing -------------------

def embed_text_bert(text):
    inputs = tokenizer_bert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def preprocess_for_seq_models(text):
    sequence = tokenizer_seq.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=148)

def normalize_symptoms(text):
    text = text.lower()
    tokens = text.split()
    normalized = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens) + 1):
            phrase = " ".join(tokens[i:j])
            if phrase in symptom_map:
                normalized.append(symptom_map[phrase])
                break
    if not normalized:
        normalized.append(text)
    return " ".join(normalized)

# -------------------- Ensemble Voting -----------------------------

def soft_voting_prediction(text):
    try:
        tfidf_vector = tfidf_vectorizer.transform([text]).toarray()
        bert_vector = np.array([embed_text_bert(text)])
        seq_vector = preprocess_for_seq_models(text)

        # Keras models use .predict()
        proba_bert = models['biobert'].predict(bert_vector)[0]  # FIXED HERE
        proba_textcnn = models['TextCNN'].predict(seq_vector)[0]  # FIXED HERE
        proba_bilstm = models['BiLSTM'].predict(seq_vector)[0]  # FIXED HERE

        # Scikit-learn models use .predict_proba()
        proba_adaboost = models['adaboost'].predict_proba(tfidf_vector)[0]
        proba_rf = models['random_forest'].predict_proba(tfidf_vector)[0]
        proba_gb = models['gradient_boost'].predict_proba(tfidf_vector)[0]
        proba_mlp = models['mlp'].predict_proba(tfidf_vector)[0]

        average_proba = (
            0.2 * proba_bert +
            0.1 * proba_adaboost +
            0.1 * proba_rf +
            0.1 * proba_gb +
            0.15 * proba_mlp +
            0.175 * proba_textcnn +
            0.175 * proba_bilstm
        )

        top_k_indices = average_proba.argsort()[-3:][::-1]
        results = [
            {'disease': disease_labels[i], 'confidence': round(float(average_proba[i]) * 100, 2)}
            for i in top_k_indices
        ]
        return results
    except Exception as e:
        traceback.print_exc()
        return {'error': f'Ensemble prediction failed: {str(e)}'}

# ------------------------ Routes ----------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin@example.com' and password == '1234':
            return redirect(url_for('interface'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        user = request.form.get('email')
        pswd = request.form.get('password')
        return redirect(url_for('interface'))
    return render_template('signup.html', error=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/interface')
def interface():
    return render_template('interface.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'model' not in data:
            return jsonify({'error': 'Invalid input'}), 400

        input_text = normalize_symptoms(data['text'])
        model_name = data['model']

        if model_name == 'ensemble':
            results = soft_voting_prediction(input_text)
            if isinstance(results, dict) and 'error' in results:
                return jsonify(results), 500
            return jsonify({'top_k_predictions': results})

        if model_name not in models:
            return jsonify({'error': 'Invalid model selected'}), 400

        model = models[model_name]

        if model_name == 'biobert':
            input_vector = np.array([embed_text_bert(input_text)])
        elif model_name in ['RNN', 'TextCNN', 'BiLSTM']:
            input_vector = preprocess_for_seq_models(input_text)
        elif model_name in ['adaboost', 'random_forest', 'gradient_boost', 'mlp']:
            input_vector = tfidf_vectorizer.transform([input_text]).toarray()
        else:
            return jsonify({'error': 'Unsupported model type'}), 400

        # FIXED: Use predict_proba only for sklearn models; Keras models use predict
        if model_name in ['adaboost', 'random_forest', 'gradient_boost', 'mlp']:
            prediction_raw = model.predict_proba(input_vector)
        else:
            prediction_raw = model.predict(input_vector)

        if isinstance(prediction_raw, (int, float)) or np.ndim(prediction_raw) == 0:
            raise ValueError("Model returned scalar instead of probability vector")

        prediction = prediction_raw[0].astype(float)
        top_k_indices = prediction.argsort()[-3:][::-1]
        results = [
            {'disease': disease_labels[i], 'confidence': round(float(prediction[i]) * 100, 2)}
            for i in top_k_indices
        ]

        return jsonify({'top_k_predictions': results})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# ------------------- Launch App ------------------------

if __name__ == '__main__':
    app.run(debug=True)
# To run the app, use the command: python app.py