"""
This script trains the final DILI prediction model on the entire
processed dataset and saves the trained model to a file for later use.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import ast

PROCESSED_DATA_PATH = 'data/processed/dili_data_clean.csv'
MODEL_OUTPUT_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'random_forest_dili_model.pkl')

def train_final_model(data_path, model_save_path):
    """
    Loads the processed data, trains a RandomForestClassifier, and saves the model.

    Args:
        data_path (str): Path to the processed CSV file.
        model_save_path (str): Path to save the final trained model.
    """
    print("--- Starting Model Training ---")

    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded processed data from '{data_path}'.")
    except FileNotFoundError:
        print(f"Error: Processed data file not found at '{data_path}'.")
        print("Please run the data_processing.py script first.")
        return

    # Prepare Data for Modeling 
    # Drop any remaining rows with missing fingerprints
    df.dropna(subset=['fingerprint'], inplace=True)
    
    # Convert the string representation of the fingerprint list back into a list
    df['fingerprint'] = df['fingerprint'].apply(ast.literal_eval)
    
    # Separate features (X) and target (y)
    X = np.array(df['fingerprint'].tolist())
    y = df['dili_concern'].values
    
    print(f"Data prepared for training. Feature shape: {X.shape}, Target shape: {y.shape}")

    # Initialize and Train the Model
    # We use the parameters that gave us our best baseline performance.
    print("Initializing RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,          # The tuned parameter we found
        class_weight='balanced',   # Handles class imbalance
        random_state=42,           # For reproducibility
        n_jobs=-1                  # Use all available CPU cores
    )
    
    print("Training model on the full dataset...")
    model.fit(X, y)
    print("Model training complete.")

    # Save the Trained Model
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Save the model object to a file using joblib
    joblib.dump(model, model_save_path)
    print(f"Model successfully saved to '{model_save_path}'")

if __name__ == '__main__':
    # This block allows the script to be run directly from the command line
    train_final_model(PROCESSED_DATA_PATH, MODEL_PATH)
