#!/usr/bin/env python3
"""
SageMaker training script for AI Scientist Team experiments
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import json


def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Hyperparameters
    parser.add_argument('--model_type', type=str, default='random_forest')
    parser.add_argument('--target_variable', type=str, default='target')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    
    return parser.parse_args()


def load_data(train_path):
    """Load data from SageMaker input path"""
    # Look for parquet files first, then CSV
    for file in os.listdir(train_path):
        if file.endswith('.parquet'):
            return pd.read_parquet(os.path.join(train_path, file))
        elif file.endswith('.csv'):
            return pd.read_csv(os.path.join(train_path, file))
    
    raise ValueError("No supported data files found in training path")


def create_model(model_type, task_type):
    """Create model based on type and task"""
    if task_type == 'classification':
        if model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            return LogisticRegression(random_state=42, max_iter=1000)
    else:  # regression
        if model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'linear_regression':
            return LinearRegression()
    
    # Default to random forest classifier
    return RandomForestClassifier(n_estimators=100, random_state=42)


def main():
    args = parse_args()
    
    # Load data
    print("Loading training data...")
    df = load_data(args.train)
    print(f"Data shape: {df.shape}")
    
    # Determine target column
    target_col = args.target_variable
    if target_col not in df.columns:
        # Use last column as target
        target_col = df.columns[-1]
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical target if needed
    label_encoder = None
    task_type = 'classification'
    if y.dtype == 'object' or len(np.unique(y)) < 20:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        task_type = 'classification'
    else:
        task_type = 'regression'
    
    print(f"Task type: {task_type}")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {target_col}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    
    # Create and train model
    print(f"Training {args.model_type} model...")
    model = create_model(args.model_type, task_type)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    if task_type == 'classification':
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['precision'] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['recall'] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        metrics['cv_mean'] = float(cv_scores.mean())
        metrics['cv_std'] = float(cv_scores.std())
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['r2_score'] = float(r2_score(y_test, y_pred))
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        metrics['cv_mean'] = float(cv_scores.mean())
        metrics['cv_std'] = float(cv_scores.std())
    
    print(f"Model metrics: {metrics}")
    
    # Feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_importance = {
            name: float(importance) 
            for name, importance in zip(X.columns, model.feature_importances_)
        }
    
    # Save model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save label encoder if used
    if label_encoder:
        encoder_path = os.path.join(args.model_dir, 'label_encoder.joblib')
        joblib.dump(label_encoder, encoder_path)
    
    # Save results
    results = {
        'model_type': args.model_type,
        'task_type': task_type,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'data_shape': list(df.shape),
        'feature_names': list(X.columns),
        'target_name': target_col
    }
    
    results_path = os.path.join(args.output_data_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()