import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import pickle

# Load the cleaned dataset
print("Loading cleaned data...")
data = pd.read_csv('cleaned_data.csv')

# Split into features and labels
print("Splitting data into features and labels...")
X = data['comment_text']
y = data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# Convert text to TF-IDF vectors
print("Converting text to TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Calculate class weights (to handle imbalanced data)
print("Calculating class weights...")
class_weights = []
for col in y.columns:
    positive_samples = y[col].sum()
    total_samples = len(y)
    weight = total_samples / (2 * positive_samples)  # Assign higher weight to minority classes
    class_weights.append({0: 1, 1: weight})  # Weight for each class (0 and 1)

# Train a multi-output Logistic Regression model with class weights
print("Training the model...")
model = MultiOutputClassifier(LogisticRegression(class_weight='balanced'))  # Use class weights
model.fit(X_tfidf, y)

# Save the model and vectorizer
print("Saving the model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model training complete. Model and vectorizer saved.")