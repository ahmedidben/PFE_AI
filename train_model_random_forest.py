import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import pickle

# Load data
df = pd.read_csv('cleaned_medical_symptoms.csv')

# Vectorize text with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['cleaned_text'])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['label'].astype(str))

# Save TF-IDF vectorizer and label encoder
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
joblib.dump(le, 'label_encoder.pkl')
pd.Series(le.classes_).to_csv('disease_classes.csv', index=False, header=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build and train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'random_forest_model.pkl')

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



# Evaluation (optional)
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("Random Forest model trained and saved successfully!")