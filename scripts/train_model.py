import pandas as pd
from sklearn.metrics import classification_report
import pickle
import numpy as np

# Load the cleaned dataset
print("Loading cleaned data...")
data = pd.read_csv('cleaned_data.csv')

# Split into features and labels
print("Splitting data into features and labels...")
X = data['comment_text']
y = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# Load the model and vectorizer
print("Loading model and vectorizer...")
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Convert text to TF-IDF vectors
print("Converting text to TF-IDF vectors...")
X_tfidf = vectorizer.transform(X)

# Get predicted probabilities
print("Getting predicted probabilities...")
y_pred_proba = model.predict_proba(X_tfidf)

# Adjust the classification threshold (e.g., 0.3)
threshold = 0.3
y_pred = (np.array(y_pred_proba) > threshold).astype(int)

# Flatten the predictions to match the shape of y
y_pred = np.hstack(y_pred)

# Evaluate the model
print("Evaluating the model...")
print(classification_report(y, y_pred, target_names=y.columns))