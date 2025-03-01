import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('train.csv')

# Drop rows with missing or invalid 'comment_text'
print("Dropping missing or invalid rows...")
data = data.dropna(subset=['comment_text'])  # Drop rows where 'comment_text' is NaN
data = data[data['comment_text'].astype(bool)]  # Drop rows with empty strings

# Check for any remaining NaN values in 'comment_text'
if data['comment_text'].isnull().any():
    print("Warning: NaN values still exist in 'comment_text'. Dropping them...")
    data = data.dropna(subset=['comment_text'])

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):  # Check if the text is not a string
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    text = ' '.join([word for word in tokens if word not in stop_words])
    return text

# Apply cleaning to the comment text
print("Cleaning text...")
data['comment_text'] = data['comment_text'].apply(clean_text)

# Check for any empty strings after cleaning
print("Checking for empty strings after cleaning...")
data = data[data['comment_text'].astype(bool)]  # Drop rows with empty strings

# Save the cleaned data
print("Saving cleaned data...")
data.to_csv('cleaned_data.csv', index=False)
print("Data preprocessing complete. Cleaned data saved as 'cleaned_data.csv'.")