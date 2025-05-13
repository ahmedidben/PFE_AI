import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense

# Load cleaned data
df = pd.read_csv('cleaned_medical_symptoms.csv')

# Load BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model_bert = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Embed text
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply BioBERT embedding
X = np.vstack(df['cleaned_text'].apply(embed_text).to_numpy())

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'])
pd.Series(le.classes_).to_csv('disease_labels.csv', index=False)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(128, input_shape=(X.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# Save model
save_model(model, 'best_model2.keras')
