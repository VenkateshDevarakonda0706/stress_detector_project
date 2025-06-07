# File: utils.py
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    if not isinstance(text, str):  # Fix: only process strings
        return ""
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(cleaned)

def get_suggestion(label):
    suggestions = {
        'stress': "Take a deep breath. Try a 5-minute mindfulness exercise.",
        'sad': "Talk to a close friend. Journaling might help you express more.",
        'happy': "Great to hear! Keep spreading the joy."
    }
    return suggestions.get(label, "Stay strong. Take care of yourself.")