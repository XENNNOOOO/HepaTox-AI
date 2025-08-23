"""
This script trains the final DILI prediction ensemble model on the entire
processed dataset and saves the trained models to a single file for later use.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import os
import ast

PROCESSED_DATA_PATH = 'data/processed/dili_data_clean.csv'
MODEL_OUTPUT_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_dili_model.pkl')

def train_final_ensemble_model(data_path, model_save_path):
    """
    Loads the processed data, trains both the RandomForest and XGBoost models,
    and saves them together in a dictionary.

    Args:
        data_path (str): Path to the processed CSV file.
        model_save_path (str): Path to save the final ensemble model file.
    """
    print("--- Starting Ensemble Model Training ---")

    # Load Data
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded processed data from '{data_path}'.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{data_path}'.")
        print("Please run the data_processing.py script first.")
        return

    # Prepare Data for Modeling
    df.dropna(subset=['fingerprint'], inplace=True)
    df['fingerprint'] = df['fingerprint'].apply(ast.literal_eval)
    X = np.array(df['fingerprint'].tolist())
    y = df['dili_concern'].values
    print(f"Data prepared for training. Feature shape: {X.shape}, Target shape: {y.shape}")

    # Train RandomForest Model
    print("\nTraining RandomForest model on the full dataset...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)
    print("RandomForest training complete.")

    # Train XGBoost Model
    print("\nTraining XGBoost model on the full dataset...")
    best_xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'use_label_encoder': False,
        'random_state': 42, 'n_estimators': 100, 'max_depth': 5,
        'learning_rate': 0.1, 'subsample': 0.9, 'colsample_bytree': 0.7,
        'gamma': 0.2, 'min_child_weight': 1 # Using best params from tuning
    }
    neg_count = np.sum(y == 0)
    pos_count = np.sum(y == 1)
    best_xgb_params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1
    
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(X, y)
    print("XGBoost training complete.")

    # Save the Ensemble Model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save both models in a dictionary
    ensemble_models = {
        'random_forest': rf_model,
        'xgboost': xgb_model
    }
    
    joblib.dump(ensemble_models, model_save_path)
    print(f"\nEnsemble model successfully saved to '{model_save_path}'")

if __name__ == '__main__':
    train_final_ensemble_model(PROCESSED_DATA_PATH, MODEL_PATH)
