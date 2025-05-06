"""
Feature Extraction Module for ChatGPT Reviews Sentiment Analysis
---------------------------------------------------------------
This module handles text vectorization and feature extraction from preprocessed reviews.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import joblib

class FeatureExtractor:
    """Class for handling feature extraction tasks."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        """
        Initialize the FeatureExtractor.
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features (vocabulary size).
        ngram_range : tuple
            Range of n-grams to consider.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize vectorizers
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2  # Minimum document frequency
        )
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Apply sublinear tf scaling (1 + log(tf))
        )
        
        # For dimensionality reduction
        self.svd = TruncatedSVD(n_components=300, random_state=42)
        
        # To keep track of fitted vectorizers
        self.is_fitted = {
            'count': False,
            'tfidf': False,
            'svd': False
        }
    
    def fit_count_vectorizer(self, X_train):
        """
        Fit the Count Vectorizer on training data.
        
        Parameters:
        -----------
        X_train : array-like
            Training text data.
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Vectorized training data.
        """
        print("Fitting Count Vectorizer...")
        X_train_counts = self.count_vectorizer.fit_transform(X_train)
        self.is_fitted['count'] = True
        print(f"Count Vectorizer fitted. Shape: {X_train_counts.shape}")
        return X_train_counts
    
    def transform_count_vectorizer(self, X):
        """
        Transform data using the fitted Count Vectorizer.
        
        Parameters:
        -----------
        X : array-like
            Text data to transform.
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Vectorized data.
        """
        if not self.is_fitted['count']:
            raise ValueError("Count Vectorizer is not fitted yet.")
        
        return self.count_vectorizer.transform(X)
    
    def fit_tfidf_vectorizer(self, X_train):
        """
        Fit the TF-IDF Vectorizer on training data.
        
        Parameters:
        -----------
        X_train : array-like
            Training text data.
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Vectorized training data.
        """
        print("Fitting TF-IDF Vectorizer...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        self.is_fitted['tfidf'] = True
        print(f"TF-IDF Vectorizer fitted. Shape: {X_train_tfidf.shape}")
        return X_train_tfidf
    
    def transform_tfidf_vectorizer(self, X):
        """
        Transform data using the fitted TF-IDF Vectorizer.
        
        Parameters:
        -----------
        X : array-like
            Text data to transform.
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            Vectorized data.
        """
        if not self.is_fitted['tfidf']:
            raise ValueError("TF-IDF Vectorizer is not fitted yet.")
        
        return self.tfidf_vectorizer.transform(X)
    
    def fit_svd(self, X_train_tfidf):
        """
        Fit SVD for dimensionality reduction on the TF-IDF vectors.
        
        Parameters:
        -----------
        X_train_tfidf : scipy.sparse.csr_matrix
            TF-IDF vectors of training data.
            
        Returns:
        --------
        numpy.ndarray
            Reduced vectors.
        """
        print("Fitting SVD for dimensionality reduction...")
        X_train_svd = self.svd.fit_transform(X_train_tfidf)
        self.is_fitted['svd'] = True
        print(f"SVD fitted. Explained variance ratio: {self.svd.explained_variance_ratio_.sum():.4f}")
        print(f"Reduced shape: {X_train_svd.shape}")
        return X_train_svd
    
    def transform_svd(self, X_tfidf):
        """
        Transform TF-IDF vectors using the fitted SVD.
        
        Parameters:
        -----------
        X_tfidf : scipy.sparse.csr_matrix
            TF-IDF vectors to transform.
            
        Returns:
        --------
        numpy.ndarray
            Reduced vectors.
        """
        if not self.is_fitted['svd']:
            raise ValueError("SVD is not fitted yet.")
        
        return self.svd.transform(X_tfidf)
    
    def get_top_features(self, n=20):
        """
        Get the top n features (words) from the TF-IDF vectorizer.
        
        Parameters:
        -----------
        n : int
            Number of top features to return.
            
        Returns:
        --------
        list
            List of top n features.
        """
        if not self.is_fitted['tfidf']:
            raise ValueError("TF-IDF Vectorizer is not fitted yet.")
        
        names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Sum the TF-IDF values for each feature to get the overall importance
        # We need a document to do this, so we'll create a dummy one
        dummy_doc = " ".join(names)
        response = self.tfidf_vectorizer.transform([dummy_doc])
        
        feature_importance = np.asarray(response.sum(axis=0)).flatten()
        indices = np.argsort(feature_importance)[::-1][:n]
        
        return [(names[i], feature_importance[i]) for i in indices]
    
    def save_vectorizers(self, path_prefix='../models/'):
        """
        Save trained vectorizers and SVD to disk.
        
        Parameters:
        -----------
        path_prefix : str
            Path prefix for saving models.
        """
        if self.is_fitted['count']:
            joblib.dump(self.count_vectorizer, f"{path_prefix}count_vectorizer.joblib")
            print(f"Count Vectorizer saved to {path_prefix}count_vectorizer.joblib")
        
        if self.is_fitted['tfidf']:
            joblib.dump(self.tfidf_vectorizer, f"{path_prefix}tfidf_vectorizer.joblib")
            print(f"TF-IDF Vectorizer saved to {path_prefix}tfidf_vectorizer.joblib")
        
        if self.is_fitted['svd']:
            joblib.dump(self.svd, f"{path_prefix}svd.joblib")
            print(f"SVD model saved to {path_prefix}svd.joblib")
    
    @classmethod
    def load_vectorizers(cls, path_prefix='../models/'):
        """
        Load trained vectorizers and SVD from disk.
        
        Parameters:
        -----------
        path_prefix : str
            Path prefix for loading models.
            
        Returns:
        --------
        FeatureExtractor
            An instance with loaded vectorizers.
        """
        instance = cls()
        
        try:
            instance.count_vectorizer = joblib.load(f"{path_prefix}count_vectorizer.joblib")
            instance.is_fitted['count'] = True
            print(f"Count Vectorizer loaded from {path_prefix}count_vectorizer.joblib")
        except FileNotFoundError:
            print("Count Vectorizer file not found.")
        
        try:
            instance.tfidf_vectorizer = joblib.load(f"{path_prefix}tfidf_vectorizer.joblib")
            instance.is_fitted['tfidf'] = True
            print(f"TF-IDF Vectorizer loaded from {path_prefix}tfidf_vectorizer.joblib")
        except FileNotFoundError:
            print("TF-IDF Vectorizer file not found.")
        
        try:
            instance.svd = joblib.load(f"{path_prefix}svd.joblib")
            instance.is_fitted['svd'] = True
            print(f"SVD model loaded from {path_prefix}svd.joblib")
        except FileNotFoundError:
            print("SVD model file not found.")
        
        return instance

if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load the data splits
    with open('../data/data_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    X_train = splits['X_train']
    X_val = splits['X_val']
    X_test = splits['X_test']
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(max_features=5000)
    
    # Fit and transform training data
    X_train_tfidf = feature_extractor.fit_tfidf_vectorizer(X_train)
    X_train_svd = feature_extractor.fit_svd(X_train_tfidf)
    
    # Transform validation and test data
    X_val_tfidf = feature_extractor.transform_tfidf_vectorizer(X_val)
    X_val_svd = feature_extractor.transform_svd(X_val_tfidf)
    
    X_test_tfidf = feature_extractor.transform_tfidf_vectorizer(X_test)
    X_test_svd = feature_extractor.transform_svd(X_test_tfidf)
    
    # Save vectorizers
    feature_extractor.save_vectorizers()
    
    # Save vectorized data
    vectorized_data = {
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'X_train_svd': X_train_svd,
        'X_val_svd': X_val_svd,
        'X_test_svd': X_test_svd,
        'y_train': splits['y_train'],
        'y_val': splits['y_val'],
        'y_test': splits['y_test']
    }
    
    with open('../data/vectorized_data.pkl', 'wb') as f:
        pickle.dump(vectorized_data, f)
    
    print("\nSaved vectorized data to '../data/vectorized_data.pkl'")
    
    # Print top features
    print("\nTop 20 features:")
    top_features = feature_extractor.get_top_features(20)
    for word, importance in top_features:
        print(f"{word}: {importance:.4f}")
