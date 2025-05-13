import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
import random

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load data
df = pd.read_csv('cleaned_medical_symptoms.csv')
texts = df['cleaned_text'].astype(str).tolist()
labels = df['label'].astype(str).tolist()

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Embedding function
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generate embeddings
X = np.vstack([embed_text(t) for t in texts])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
pd.Series(le.classes_).to_csv('disease_labels.csv', index=False, header=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
from tensorflow.keras import Input
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=16, validation_split=0.1)

# Save model
save_model(model, 'biobert_model.keras')
print("BioBERT model trained and saved successfully!")
