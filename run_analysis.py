"""
Complete Sentiment Analysis for ChatGPT Reviews
-----------------------------------------------
This script performs the entire sentiment analysis pipeline in one file.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# File paths
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chatgpt_reviews - chatgpt_reviews.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"Using CSV file at: {CSV_PATH}")
print(f"Results will be saved to: {RESULTS_DIR}")
print(f"Models will be saved to: {MODELS_DIR}")

# Step 1: Load the data
def load_data():
    """Load the ChatGPT reviews dataset."""
    print("\n=== Loading Data ===")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset with shape: {df.shape}")
    return df

# Step 2: Clean and preprocess text
def clean_text(text, lemmatizer, stop_words):
    """Clean text by removing special characters, numbers, and converting to lowercase."""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        # Join tokens back into a string
        cleaned_text = ' '.join(cleaned_tokens)
        
        return cleaned_text
    return ""

def preprocess_data(df):
    """Preprocess the dataset."""
    print("\n=== Preprocessing Data ===")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Clean review text
    print("Cleaning review text...")
    processed_df['cleaned_review'] = processed_df['review'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    
    # Clean title text
    print("Cleaning title text...")
    processed_df['cleaned_title'] = processed_df['title'].apply(lambda x: clean_text(x, lemmatizer, stop_words))
    
    # Convert ratings to sentiment categories
    print("Converting ratings to sentiment categories...")
    processed_df['sentiment'] = processed_df['rating'].apply(lambda x: 'positive' if x >= 4 
                                         else ('negative' if x <= 2 else 'neutral'))
    
    # Convert sentiment to numerical labels
    processed_df['sentiment_label'] = processed_df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    
    return processed_df

# Step 3: Data visualization
def visualize_data(df):
    """Create and save visualizations of the data."""
    print("\n=== Visualizing Data ===")
    
    plt.figure(figsize=(15, 10))
    
    # Plot rating distribution
    plt.subplot(2, 2, 1)
    sns.countplot(x='rating', data=df)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    
    # Plot sentiment distribution
    plt.subplot(2, 2, 2)
    sns.countplot(x='sentiment', data=df)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    # Plot platform distribution
    plt.subplot(2, 2, 3)
    platform_counts = df['platform'].value_counts()
    sns.barplot(x=platform_counts.index, y=platform_counts.values)
    plt.title('Platform Distribution')
    plt.xlabel('Platform')
    plt.ylabel('Count')
    
    # Plot language distribution
    plt.subplot(2, 2, 4)
    language_counts = df['language'].value_counts().head(10)  # Top 10 languages
    sns.barplot(x=language_counts.index, y=language_counts.values)
    plt.title('Top 10 Languages')
    plt.xlabel('Language')
    plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(RESULTS_DIR, "data_distribution.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

# Step 4: Split the data
def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into training, validation, and test sets."""
    print("\n=== Splitting Data ===")
    
    # First split: training + validation vs. test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df['cleaned_review'],
        df['sentiment_label'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['sentiment_label']
    )
    
    # Second split: training vs. validation
    # Calculate validation size as a proportion of train_val
    actual_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=actual_val_size,
        random_state=random_state,
        stratify=y_train_val
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 5: Feature extraction
def extract_features(X_train, X_val, X_test):
    """Extract features using TF-IDF vectorization."""
    print("\n=== Extracting Features ===")
    
    # Initialize TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )
    
    # Fit and transform training data
    print("Fitting TF-IDF vectorizer...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    print(f"TF-IDF vectorizer fitted. Shape: {X_train_tfidf.shape}")
    
    # Transform validation and test data
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Save the vectorizer
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"Vectorizer saved to {vectorizer_path}")
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vectorizer

# Step 6: Train model
def train_model(X_train, y_train, X_val, y_val):
    """Train a Logistic Regression model."""
    print("\n=== Training Model ===")
    
    model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    print("Training Logistic Regression model...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    val_report = classification_report(y_val, y_val_pred, output_dict=True)
    print(classification_report(y_val, y_val_pred))
    
    # Save the model
    model_path = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    return model, val_report

# Step 7: Evaluate model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    print("\n=== Evaluating Model ===")
    
    # Predict on test set
    y_test_pred = model.predict(X_test)
    
    # Generate metrics
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    print(classification_report(y_test, y_test_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Negative', 'Neutral', 'Positive'],
               yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the confusion matrix plot
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    return test_report

# Step 8: Predict function (can be used for new text)
def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for a given text."""
    # Clean the text
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_text = clean_text(text, lemmatizer, stop_words)
    
    # Vectorize
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    sentiment_id = model.predict(text_tfidf)[0]
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment = sentiment_map[sentiment_id]
    
    # Get probabilities
    proba = model.predict_proba(text_tfidf)[0]
    
    return sentiment, proba

# Main function
def main():
    # Load data
    df = load_data()
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Visualize data
    visualize_data(processed_df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_df)
    
    # Extract features
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_val, X_test)
    
    # Train model
    model, val_report = train_model(X_train_tfidf, y_train, X_val_tfidf, y_val)
    
    # Evaluate model
    test_report = evaluate_model(model, X_test_tfidf, y_test)
    
    # Save processed data for later use
    processed_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed_data.csv")
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    processed_df.to_csv(processed_data_path, index=False)
    print(f"\nProcessed data saved to {processed_data_path}")
    
    # Test with some examples
    print("\n=== Testing with Examples ===")
    examples = [
        "This app is amazing! I love how intuitive it is.",
        "The service was okay, but could be better.",
        "This is the worst product I've ever used. Don't waste your money!"
    ]
    
    for example in examples:
        sentiment, probabilities = predict_sentiment(example, model, vectorizer)
        print(f"\nText: {example}")
        print(f"Predicted sentiment: {sentiment}")
        print(f"Confidence: Negative={probabilities[0]:.4f}, Neutral={probabilities[1]:.4f}, Positive={probabilities[2]:.4f}")
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to {RESULTS_DIR}")
    print(f"Models saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()
