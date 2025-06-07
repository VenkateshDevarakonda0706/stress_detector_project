# File: app.py (Flask Web Interface)
from flask import Flask, request, render_template
import joblib
from utils import preprocess, get_suggestion

app = Flask(__name__)
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    suggestion = ""
    if request.method == 'POST':
        msg = request.form['message']
        cleaned = preprocess(msg)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        result = prediction.capitalize()
        suggestion = get_suggestion(prediction)
    return render_template('index.html', result=result, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)