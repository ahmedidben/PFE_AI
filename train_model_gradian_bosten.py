import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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
pd.Series(le.classes_).to_csv('disease_classes.csv', index=False, header=False)

# Build Gradient Boosting model
model = GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinks contribution of each tree
    max_depth=3,             # Maximum depth of individual trees
    min_samples_split=2,     # Minimum samples required to split a node
    random_state=42,
    subsample=0.8            # Fraction of samples used for fitting trees
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'gradient_boosting_model.pkl')

# Evaluation
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred))

print("Gradient Boosting model trained and saved successfully!")