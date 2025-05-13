import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data
df = pd.read_csv('cleaned_medical_symptoms.csv')

# Prepare features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'].astype(str))

# Prepare labels
le = LabelEncoder()
y = le.fit_transform(df['label'].astype(str))

# Save artifacts
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Build model
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),  # Changed from base_estimator to estimator
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Save model
joblib.dump(model, 'adaboost_model.pkl')

print("Model trained and saved successfully!")