# Stress Detection from Text Messages

## 📌 Objective

Analyze user messages to detect emotional tone (e.g., neutral, happy, sad, angry, etc.) and provide supportive mental health suggestions.

---

## 📂 Dataset

The dataset used is a customized version derived from the **DailyDialog** dataset, named `dailydialog_emotions.csv`. It contains conversational sentences annotated with one of the following emotion labels:

- `neutral`
- `happy`
- `sad`
- `angry`

### 🔍 Structure:

Each row in the dataset has:

- **text**: A user message (sentence)
- **emotion**: Corresponding emotion label

Example:

| text                            | emotion |
| ------------------------------- | ------- |
| I am feeling really down today. | sad     |
| That sounds great! Let's do it. | happy   |
| Why are you always late?        | angry   |

### 🧹 Preprocessing:

- Lowercasing
- Removal of punctuation and stopwords
- Lemmatization (using NLTK)

### 🧠 Usage:

This dataset is vectorized using **TF-IDF**, then passed into a **Logistic Regression classifier** to predict emotion from input text.

📁 Dataset location: `./dailydialog_emotions.csv`

---

## 🚀 Features

- Emotion-labeled `dailydialog_emotions.csv` Dataset (from DailyDialog)
- Text Cleaning with NLTK
- TF-IDF Vectorization
- Logistic Regression Classifier
- Flask Web UI for Emotion Detection
- Emotion-based Suggestions

---

## 🧠 Tech Stack

- **Frontend**: HTML, CSS (via Flask Templates)
- **Backend**: Python (Flask)
- **ML/NLP**: Scikit-learn, NLTK
- **Deployment Ready**: Easy to run locally (support for cloud deployment)

---

## 💡 How to Run

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Train the model
python model_train.py

# Start the Flask app
python app.py

```

## 💡 Quick run

```
venv\Scripts\activate
python app.py

```

---

## 💡 To Deactivate

```
deactivate
```

---

## 🔮 Future Scope

- Add more emotion classes like **anxiety**, **fear**, and **surprise**
- Use **BERT-based transformer models** for better accuracy
- Integrate **multilingual support** for Hindi, Telugu, etc.
- Add **voice input** and a **chatbot interface** for accessibility
- Deploy on **cloud platforms** like Render, Heroku, or AWS
- Create a **mobile app UI** using Flutter
