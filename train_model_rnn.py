import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load cleaned data
df = pd.read_csv('cleaned_medical_symptoms.csv')

# Check your columns – update 'cleaned_text' if needed
texts = df['cleaned_text'].astype(str).tolist()
labels = df['label'].astype(str).tolist()

# Tokenize texts
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=100)

# Save tokenizer for inference
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Optional: save class names
pd.Series(le.classes_).to_csv('disease_labels.csv', index=False, header=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build RNN model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

# Save model
save_model(model, 'rnn_model.keras')
print("RNN model trained and saved successfully!")

