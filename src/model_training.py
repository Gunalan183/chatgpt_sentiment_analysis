"""
Model Training Module for ChatGPT Reviews Sentiment Analysis
------------------------------------------------------------
This module handles the training, evaluation, and comparison of different models.
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb

class ModelTrainer:
    """Class for training and evaluating models."""
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.models = {}
        self.scores = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
    
    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None, 
                                 grid_search=False, class_weight=None):
        """
        Train a Logistic Regression model.
        
        Parameters:
        -----------
        X_train : array-like
            Training feature vectors.
        y_train : array-like
            Training labels.
        X_val : array-like, optional
            Validation feature vectors.
        y_val : array-like, optional
            Validation labels.
        grid_search : bool
            Whether to perform grid search for hyperparameter tuning.
        class_weight : dict or 'balanced', optional
            Class weights to handle imbalanced data.
            
        Returns:
        --------
        sklearn.linear_model.LogisticRegression
            Trained model.
        """
        print("\nTraining Logistic Regression model...")
        start_time = time()
        
        if grid_search and X_val is not None and y_val is not None:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
            
            model = GridSearchCV(
                LogisticRegression(class_weight=class_weight, random_state=42),
                param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            print(f"Best parameters: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                max_iter=1000,
                class_weight=class_weight,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        training_time = time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            print(f"Validation F1 Score: {score:.4f}")
            
            # Store the model and score
            self.models['logistic_regression'] = model
            self.scores['logistic_regression'] = score
            
            # Update best model if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_model_name = 'logistic_regression'
        
        return model
    
    def train_naive_bayes(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train a Multinomial Naive Bayes model.
        
        Parameters:
        -----------
        X_train : array-like
            Training feature vectors.
        y_train : array-like
            Training labels.
        X_val : array-like, optional
            Validation feature vectors.
        y_val : array-like, optional
            Validation labels.
            
        Returns:
        --------
        sklearn.naive_bayes.MultinomialNB
            Trained model.
        """
        print("\nTraining Naive Bayes model...")
        start_time = time()
        
        model = MultinomialNB(alpha=1.0)
        model.fit(X_train, y_train)
        
        training_time = time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            print(f"Validation F1 Score: {score:.4f}")
            
            # Store the model and score
            self.models['naive_bayes'] = model
            self.scores['naive_bayes'] = score
            
            # Update best model if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_model_name = 'naive_bayes'
        
        return model
    
    def train_svm(self, X_train, y_train, X_val=None, y_val=None, 
                 grid_search=False, class_weight=None):
        """
        Train a Linear Support Vector Machine model.
        
        Parameters:
        -----------
        X_train : array-like
            Training feature vectors.
        y_train : array-like
            Training labels.
        X_val : array-like, optional
            Validation feature vectors.
        y_val : array-like, optional
            Validation labels.
        grid_search : bool
            Whether to perform grid search for hyperparameter tuning.
        class_weight : dict or 'balanced', optional
            Class weights to handle imbalanced data.
            
        Returns:
        --------
        sklearn.svm.LinearSVC
            Trained model.
        """
        print("\nTraining SVM model...")
        start_time = time()
        
        if grid_search and X_val is not None and y_val is not None:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'loss': ['hinge', 'squared_hinge'],
                'max_iter': [2000]
            }
            
            model = GridSearchCV(
                LinearSVC(dual=False, class_weight=class_weight, random_state=42),
                param_grid,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            print(f"Best parameters: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = LinearSVC(
                C=1.0,
                loss='squared_hinge',
                dual=False,
                max_iter=2000,
                class_weight=class_weight,
                random_state=42
            )
            model.fit(X_train, y_train)
        
        training_time = time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            print(f"Validation F1 Score: {score:.4f}")
            
            # Store the model and score
            self.models['svm'] = model
            self.scores['svm'] = score
            
            # Update best model if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_model_name = 'svm'
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_val=None, y_val=None, 
                           grid_search=False, class_weight=None):
        """
        Train a Random Forest model.
        
        Parameters:
        -----------
        X_train : array-like
            Training feature vectors.
        y_train : array-like
            Training labels.
        X_val : array-like, optional
            Validation feature vectors.
        y_val : array-like, optional
            Validation labels.
        grid_search : bool
            Whether to perform grid search for hyperparameter tuning.
        class_weight : dict or 'balanced', optional
            Class weights to handle imbalanced data.
            
        Returns:
        --------
        sklearn.ensemble.RandomForestClassifier
            Trained model.
        """
        print("\nTraining Random Forest model...")
        start_time = time()
        
        if grid_search and X_val is not None and y_val is not None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            model = RandomizedSearchCV(
                RandomForestClassifier(class_weight=class_weight, random_state=42),
                param_grid,
                n_iter=10,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            print(f"Best parameters: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        training_time = time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            print(f"Validation F1 Score: {score:.4f}")
            
            # Store the model and score
            self.models['random_forest'] = model
            self.scores['random_forest'] = score
            
            # Update best model if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_model_name = 'random_forest'
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, 
                     grid_search=False):
        """
        Train an XGBoost model.
        
        Parameters:
        -----------
        X_train : array-like
            Training feature vectors.
        y_train : array-like
            Training labels.
        X_val : array-like, optional
            Validation feature vectors.
        y_val : array-like, optional
            Validation labels.
        grid_search : bool
            Whether to perform grid search for hyperparameter tuning.
            
        Returns:
        --------
        xgboost.XGBClassifier
            Trained model.
        """
        print("\nTraining XGBoost model...")
        start_time = time()
        
        if grid_search and X_val is not None and y_val is not None:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            model = RandomizedSearchCV(
                xgb.XGBClassifier(objective='multi:softmax', random_state=42),
                param_grid,
                n_iter=10,
                cv=3,
                scoring='f1_weighted',
                n_jobs=-1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            print(f"Best parameters: {model.best_params_}")
            model = model.best_estimator_
        else:
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softmax',
                random_state=42
            )
            model.fit(X_train, y_train)
        
        training_time = time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            print(f"Validation F1 Score: {score:.4f}")
            
            # Store the model and score
            self.models['xgboost'] = model
            self.scores['xgboost'] = score
            
            # Update best model if this is better
            if score > self.best_score:
                self.best_score = score
                self.best_model = model
                self.best_model_name = 'xgboost'
        
        return model
    
    def evaluate_model(self, model, X, y, model_name=None):
        """
        Evaluate a model on given data.
        
        Parameters:
        -----------
        model : classifier
            Trained model.
        X : array-like
            Feature vectors.
        y : array-like
            True labels.
        model_name : str, optional
            Name of the model for reporting.
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics.
        """
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        # Generate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Detailed classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        # Print results
        name = model_name if model_name else "Model"
        print(f"\nEvaluation metrics for {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        return metrics
    
    def plot_confusion_matrix(self, cm, labels, model_name=None, save_path=None):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        cm : array-like
            Confusion matrix.
        labels : list
            List of class labels.
        model_name : str, optional
            Name of the model for the title.
        save_path : str, optional
            Path to save the plot.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        
        title = f"Confusion Matrix - {model_name}" if model_name else "Confusion Matrix"
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, save_path=None):
        """
        Plot comparison of model performances.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot.
        """
        if not self.scores:
            print("No models have been trained and evaluated yet.")
            return
        
        models = list(self.scores.keys())
        f1_scores = [self.scores[model] for model in models]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=models, y=f1_scores)
        
        plt.title('Model Comparison - F1 Scores')
        plt.xlabel('Model')
        plt.ylabel('F1 Score (Validation)')
        plt.ylim(0, 1.0)
        
        # Add the actual scores as text
        for i, score in enumerate(f1_scores):
            plt.text(i, score + 0.02, f"{score:.4f}", ha='center')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_models(self, path_prefix='../models/'):
        """
        Save all trained models to disk.
        
        Parameters:
        -----------
        path_prefix : str
            Path prefix for saving models.
        """
        for name, model in self.models.items():
            try:
                joblib.dump(model, f"{path_prefix}{name}_model.joblib")
                print(f"Model '{name}' saved to {path_prefix}{name}_model.joblib")
            except Exception as e:
                print(f"Error saving model '{name}': {e}")
        
        # Save best model separately
        if self.best_model is not None:
            try:
                joblib.dump(self.best_model, f"{path_prefix}best_model.joblib")
                
                # Save best model name
                with open(f"{path_prefix}best_model_name.txt", 'w') as f:
                    f.write(self.best_model_name)
                
                print(f"Best model ('{self.best_model_name}') saved to {path_prefix}best_model.joblib")
            except Exception as e:
                print(f"Error saving best model: {e}")
    
    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the model file.
            
        Returns:
        --------
        object
            Loaded model.
        """
        try:
            model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load vectorized data
    with open('../data/vectorized_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train_tfidf']  # or X_train_svd
    X_val = data['X_val_tfidf']      # or X_val_svd
    X_test = data['X_test_tfidf']    # or X_test_svd
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    trainer.train_naive_bayes(X_train, y_train, X_val, y_val)
    trainer.train_svm(X_train, y_train, X_val, y_val)
    trainer.train_random_forest(X_train, y_train, X_val, y_val)
    trainer.train_xgboost(X_train, y_train, X_val, y_val)
    
    # Print best model
    print(f"\nBest model: {trainer.best_model_name} with F1 Score: {trainer.best_score:.4f}")
    
    # Plot model comparison
    trainer.plot_model_comparison(save_path='../results/model_comparison.png')
    
    # Evaluate best model on test set
    if trainer.best_model is not None:
        print("\nEvaluating best model on test set...")
        metrics = trainer.evaluate_model(trainer.best_model, X_test, y_test, trainer.best_model_name)
        
        # Plot confusion matrix
        class_labels = ['Negative', 'Neutral', 'Positive']
        trainer.plot_confusion_matrix(
            metrics['confusion_matrix'], 
            class_labels, 
            trainer.best_model_name,
            save_path=f'../results/{trainer.best_model_name}_confusion_matrix.png'
        )
    
    # Save all models
    trainer.save_models()
