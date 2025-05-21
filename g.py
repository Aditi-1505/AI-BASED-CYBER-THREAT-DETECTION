import pandas as pd
import numpy as np
import re
import string
from urllib.parse import urlparse
import tldextract

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------- Setup NLTK --------------------
try:
    nltk.download('stopwords')
except:
    print("Warning: NLTK stopwords download failed.")

# -------------------- URL Classification --------------------
def extract_features(url):
    parsed_url = urlparse(url)
    ext = tldextract.extract(url)
    return {
        'url_length': len(url),
        'num_special_chars': sum(1 for char in url if char in ['$', '#', '@', '&', '=', '?']),
        'num_dots': url.count('.'),
        'has_https': 1 if parsed_url.scheme == 'https' else 0,
        'num_path_segments': len(parsed_url.path.split('/')) - 1,
        'has_suspicious_keywords': int(any(k in url.lower() for k in ['login', 'pay', 'account'])),
        'domain_length': len(ext.domain),
        'suffix_length': len(ext.suffix)
    }

def train_url_classifier(file_path):
    df = pd.read_csv(file_path)
    # Output preview and column names
    print("=== URL Dataset Preview ===")
    print(df.head())
    print(f"\nColumn names in the DataFrame: {df.columns.tolist()}\n")
    if 'type' not in df.columns:
        raise KeyError("Missing 'type' column.")
    df = df.dropna(subset=['type'])

    df['label'] = df['type'].apply(lambda x: 1 if x == 'malicious' else 0)
    features = df['url'].apply(extract_features)
    X = pd.DataFrame(features.tolist())
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("=== URL Classification Results ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model


def predict_url(model, url):
    features = extract_features(url)
    prediction = model.predict(pd.DataFrame([features]))
    return "Malicious" if prediction[0] == 1 else "Non-Malicious"

# -------------------- Email Clustering --------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()

    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = set([
            'the', 'and', 'is', 'in', 'to', 'of', 'for', 'on', 'with', 'as', 'at',
            'this', 'that', 'a', 'an', 'it', 'are', 'was', 'be', 'from', 'or', 'by',
            'you', 'your', 'have', 'has', 'will', 'not', 'can', 'we', 'i', 'he', 'she',
            'they', 'them', 'my', 'me', 'but', 'if', 'about', 'into', 'out', 'what',
            'so', 'just', 'up', 'down', 'no', 'yes', 'then', 'there', 'been', 'do', 'did'
        ])

    words = [w for w in text.split() if w not in stop_words]
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    return ' '.join(stemmed)

def cluster_emails(file_path):
    df = pd.read_csv(file_path)
    if 'email_text' in df.columns:
        df['email'] = df['email_text']
    elif 'text' in df.columns:
        df['email'] = df['text']
    else:
        raise ValueError("CSV must contain 'email_text' or 'text'.")

    df['clean_email'] = df['email'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_email'])

    kmeans = KMeans(n_clusters=2, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    score = silhouette_score(X, df['cluster'])
    print("\n=== Email Clustering ===")
    print(f"Silhouette Score: {score:.2f}")

    for cluster in range(2):
        print(f"\nCluster {cluster} sample emails:")
        print(df[df['cluster'] == cluster]['email'].head(3).to_string(index=False))

    df['label'] = df['cluster'].apply(lambda x: 'malicious' if x == 1 else 'benign')

    print("\nFinal Output Sample:")
    print(df[['email', 'label']].head(10))

    return df

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    url_csv_path = '/Users/suryapalsinghbisht/Downloads/malicious_phish.csv'
    email_csv_path = '/Users/suryapalsinghbisht/Downloads/emails.csv'

    url_model = train_url_classifier(url_csv_path)

    test_url = "linkedin.com/pub/tim-stepanski/18/319/149"
    print(f"\nPrediction for URL '{test_url}': {predict_url(url_model, test_url)}")

    cluster_emails(email_csv_path)
