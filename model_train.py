# File: model_train.py (for offline training)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from utils import preprocess

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load dataset
df = pd.read_csv('dataset.csv')
df['cleaned'] = df['text'].apply(preprocess)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Add zero_division=0 here to suppress warnings and handle zero divisions gracefully
print(classification_report(y_test, model.predict(X_test), zero_division=0))

# Save model
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')