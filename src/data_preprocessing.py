"""
Data Preprocessing Module for ChatGPT Reviews Sentiment Analysis
----------------------------------------------------------------
This module handles loading, cleaning, and preprocessing of the ChatGPT reviews dataset.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreprocessor:
    """Class for handling data preprocessing tasks."""
    
    def __init__(self, data_path):
        """
        Initialize the DataPreprocessor.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV data file.
        """
        self.data_path = data_path
        self.data = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load the dataset from CSV."""
        self.data = pd.read_csv(self.data_path)
        print(f"Loaded dataset with shape: {self.data.shape}")
        return self.data
    
    def clean_text(self, text):
        """
        Clean text by removing special characters, numbers, and converting to lowercase.
        
        Parameters:
        -----------
        text : str
            Text to clean.
            
        Returns:
        --------
        str
            Cleaned text.
        """
        if isinstance(text, str):
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
            
            # Join tokens back into a string
            cleaned_text = ' '.join(cleaned_tokens)
            
            return cleaned_text
        return ""
    
    def preprocess(self):
        """Preprocess the dataset."""
        if self.data is None:
            self.load_data()
        
        # Create a copy to avoid modifying the original
        df = self.data.copy()
        
        # Clean review text
        print("Cleaning review text...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # Clean title text
        print("Cleaning title text...")
        df['cleaned_title'] = df['title'].apply(self.clean_text)
        
        # Convert ratings to sentiment categories
        print("Converting ratings to sentiment categories...")
        df['sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 4 
                                             else ('negative' if x <= 2 else 'neutral'))
        
        # Convert sentiment to numerical labels
        df['sentiment_label'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})
        
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split data into training, validation, and test sets.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed data.
        test_size : float
            Proportion of data to be used for testing.
        val_size : float
            Proportion of training data to be used for validation.
        random_state : int
            Random seed for reproducibility.
            
        Returns:
        --------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
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
    
    def plot_distribution(self, df, save_path=None):
        """
        Plot the distribution of ratings and sentiments.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Preprocessed data.
        save_path : str, optional
            Path to save the plot.
        """
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
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()

if __name__ == "__main__":
    # Example usage
    data_path = "../../chatgpt_reviews - chatgpt_reviews.csv"
    preprocessor = DataPreprocessor(data_path)
    df = preprocessor.load_data()
    
    # Display basic info
    print("\nDataset info:")
    print(df.info())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Preprocess data
    processed_df = preprocessor.preprocess()
    
    # Save preprocessed data
    processed_df.to_csv('../data/processed_chatgpt_reviews.csv', index=False)
    print("\nSaved preprocessed data to '../data/processed_chatgpt_reviews.csv'")
    
    # Plot distributions
    preprocessor.plot_distribution(processed_df, save_path='../results/data_distribution.png')
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(processed_df)
    
    # Save splits for later use
    splits = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }
    
    import pickle
    with open('../data/data_splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    
    print("\nSaved data splits to '../data/data_splits.pkl'")
