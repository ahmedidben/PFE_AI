import streamlit as st
import json
import re
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.title("🩺 Symptom Normalization & Medical Condition Prediction")

# Load assets once
@st.cache_resource
def load_assets():
    model = load_model("best_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

model, tokenizer, le = load_assets()

# Load synonym dictionary
with open("try.json", "r") as file:
    synonym_dict = json.load(file)

# Normalize symptoms
def normalize_symptoms(text):
    text = text.lower()
    for phrase in sorted(synonym_dict.keys(), key=lambda x: -len(x.split())):
        if phrase in text:
            text = text.replace(phrase, synonym_dict[phrase])
    tokens = re.findall(r'\b\w+\b', text)
    normalized = [synonym_dict.get(word, word) for word in tokens]
    return ' '.join(normalized)

# Input
user_input = st.text_input("Enter your symptoms (e.g. 'fever and sore throat'):")

if user_input:
    normalized_input = normalize_symptoms(user_input)
    st.write("🔎 Normalized Input:", normalized_input)

    input_sequence = tokenizer.texts_to_sequences([normalized_input])
    padded_input = pad_sequences(input_sequence, maxlen=100)

    prediction = model.predict(padded_input)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_label = le.inverse_transform(predicted_class)[0]
    confidence = np.max(prediction, axis=1)[0]

    st.subheader("🧠 Prediction Result")
    st.success(f"Predicted Condition: {predicted_label}")
    st.info(f"Confidence: {confidence:.2%}")

    if st.checkbox("Show detailed probabilities"):
        prob_df = pd.DataFrame({
            "Condition": le.classes_,
            "Probability": prediction[0]
        }).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Condition"))
