import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score
import pandas as pd
import pickle
import numpy as np
import os

# Create the 'images' folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

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

# Evaluate the model at different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
precision_list = []
recall_list = []
f1_list = []

for threshold in thresholds:
    y_pred = [(y_pred_proba[i][:, 1] > threshold).astype(int) for i in range(len(model.estimators_))]
    y_pred = np.array(y_pred).T
    report = classification_report(y, y_pred, target_names=y.columns, output_dict=True, zero_division=0)
    precision_list.append(report['weighted avg']['precision'])
    recall_list.append(report['weighted avg']['recall'])
    f1_list.append(report['weighted avg']['f1-score'])

# Plot Precision-Recall Trade-off
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precision_list, marker='o', label='Precision')
plt.plot(thresholds, recall_list, marker='o', label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall Trade-off')
plt.legend()
plt.grid(True)
plt.savefig('images/precision_recall_tradeoff.png')
plt.show()

# Plot F1-Score vs. Threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_list, marker='o', color='green')
plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.title('F1-Score vs. Threshold')
plt.grid(True)
plt.savefig('images/f1_score_vs_threshold.png')
plt.show()

# Plot Class Distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=y.columns, y=y.sum(), palette='viridis')
plt.xlabel('Toxicity Category')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in the Dataset')
plt.xticks(rotation=45)
plt.savefig('images/class_distribution.png')
plt.show()

# Plot Confusion Matrix (for threshold = 0.3)
threshold = 0.3
y_pred = [(y_pred_proba[i][:, 1] > threshold).astype(int) for i in range(len(model.estimators_))]
y_pred = np.array(y_pred).T
cm = confusion_matrix(y.values.argmax(axis=1), y_pred.argmax(axis=1))

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Threshold = 0.3)')
plt.savefig('images/confusion_matrix.png')
plt.show()

# Save evaluation results to a text file
with open('evaluation_results.txt', 'w') as f:
    f.write("Evaluation Results\n")
    f.write("=================\n")
    f.write(f"Thresholds: {thresholds}\n")
    f.write(f"Precision: {precision_list}\n")
    f.write(f"Recall: {recall_list}\n")
    f.write(f"F1-Scores: {f1_list}\n")
    f.write("\nClass Distribution\n")
    f.write("=================\n")
    for col in y.columns:
        f.write(f"{col}: {y[col].sum()}\n")

print("Graphs and evaluation results saved successfully!")