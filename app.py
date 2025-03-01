from flask import Flask, request, render_template
import pickle
import numpy as np
import nltk

# Download NLTK data once (outside the route)
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Flask app
app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text cleaning function
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    threshold = float(request.form.get('threshold', 0.3))  # Default threshold is 0.3
    comment_clean = clean_text(comment)
    comment_vec = vectorizer.transform([comment_clean])
    y_pred_proba = model.predict_proba(comment_vec)
    y_pred = [(y_pred_proba[i][:, 1] > threshold).astype(int) for i in range(len(model.estimators_))]
    result = {label: pred[0] for label, pred in zip(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], y_pred)}
    return render_template('index.html', comment=comment, result=result, threshold=threshold)

if __name__ == '__main__':
    app.run(debug=True)