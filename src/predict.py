"""
This script uses the trained RandomForest model to predict the DILI
(Drug-Induced Liver Injury) risk for a new compound name provided
as a command-line argument.
"""

import joblib
import numpy as np
import argparse
from data_processing import get_smiles_from_name, generate_fingerprint

MODEL_PATH = 'models/ensemble_model.pkl'

def predict_dili_risk(compound_name, model):
    """
    Predicts the DILI risk for a single compound name.

    Args:
        compound_name (str): The name of the drug/compound.
        model: The trained machine learning model object.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and the probability,
               or (None, None) if the compound cannot be processed.
    """
    print(f"\nProcessing compound: '{compound_name}'")

    print("  - Step 1: Fetching molecular structure (SMILES)...")
    smiles = get_smiles_from_name(compound_name)
    if smiles is None:
        print("  - ERROR: Could not find a molecular structure for this compound.")
        return None, None
    print(f"  - SMILES found: {smiles}")

    print("  - Step 2: Generating molecular fingerprint...")
    fingerprint = generate_fingerprint(smiles)
    if fingerprint is None:
        print("  - ERROR: Could not generate a fingerprint from the SMILES string.")
        return None, None
    print("  - Fingerprint generated successfully.")

    # The model expects a 2D array, so we reshape our single fingerprint
    fingerprint_2d = np.array(fingerprint).reshape(1, -1)
    
    print("  - Step 3: Making a prediction with the trained model...")
    prediction = model.predict(fingerprint_2d)[0]
    probability = model.predict_proba(fingerprint_2d)[0][1] # Probability of class 1 (DILI Concern)
    
    return prediction, probability

def main():
    """
    Main function to parse command-line arguments and run the prediction.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Predict DILI risk for a given compound name.")
    parser.add_argument("compound_name", type=str, help="The name of the drug to evaluate.")
    args = parser.parse_args()

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded trained model from '{MODEL_PATH}'.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please run the train.py script first to train and save the model.")
        return

    # Run prediction
    prediction, probability = predict_dili_risk(args.compound_name, model)

    if prediction is not None:
        print("\n--- Prediction Result ---")
        if prediction == 1:
            print(f"Result: DILI Concern")
        else:
            print(f"Result: No DILI Concern")
        
        print(f"Confidence (Probability of DILI Concern): {probability:.2%}")
        print("-----------------------")

if __name__ == '__main__':
    main()
