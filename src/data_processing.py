"""
This script contains functions for loading, cleaning, and featurizing
the DILI (Drug-Induced Liver Injury) dataset. It handles the entire
pipeline from a raw CSV file to a model-ready DataFrame.
"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import cirpy
from tqdm import tqdm

# Configure tqdm to work with pandas' `progress_apply`
tqdm.pandas()

def get_smiles_from_name(compound_name):
    """
    Resolves a compound name to its canonical SMILES string using the cirpy library.

    Args:
        compound_name (str): The name of the drug/compound.

    Returns:
        str: The canonical SMILES string, or None if not found.
    """
    try:
        # Resolve the name to a SMILES string
        smiles = cirpy.resolve(compound_name, 'smiles')
        return smiles
    except Exception as e:
        print(f"An error occurred for '{compound_name}': {e}")
        return None

def generate_fingerprint(smiles):
    """
    Generates a 1024-bit Morgan Fingerprint from a SMILES string.

    Args:
        smiles (str): The SMILES string of a molecule.

    Returns:
        list: A list of 1024 integers (0s and 1s), or None if the SMILES is invalid.
    """
    if not isinstance(smiles, str):
        return None
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # Generate a 1024-bit Morgan fingerprint with a radius of 2
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return list(fp)
    else:
        return None

def process_raw_data(input_path, output_path):
    """
    Main function to run the entire data processing pipeline. It loads the raw data,
    cleans it, fetches SMILES, generates fingerprints, and saves the processed file.

    Args:
        input_path (str): The file path for the raw DILIrank CSV.
        output_path (str): The file path to save the processed CSV.
    """
    print("Starting data processing pipeline...")
    
    # Load Data 
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded raw data from '{input_path}'.")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{input_path}'.")
        return

    # Clean Data and Create Target Variable
    df['dili_concern'] = df['vDILIConcern'].apply(lambda x: 0 if str(x).strip() == 'No-DILI-Concern' else 1)
    df['Compound Name'] = df['Compound Name'].str.strip()
    df_clean = df[['Compound Name', 'vDILIConcern', 'dili_concern']].copy()
    print("Data cleaned and binary target created.")

    # Fetch SMILES Strings 
    print("Fetching SMILES strings... (This may take a while)")
    df_clean['smiles'] = df_clean['Compound Name'].progress_apply(get_smiles_from_name)
    
    # Generate Fingerprints 
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)
    
    print("Generating molecular fingerprints...")
    df_clean['fingerprint'] = df_clean['smiles'].progress_apply(generate_fingerprint)
    
    # Re-enable RDKit warnings
    lg.setLevel(RDLogger.INFO)
    
    # Drop rows where SMILES or fingerprint could not be generated
    df_final = df_clean.dropna(subset=['smiles', 'fingerprint']).copy()
    
    num_dropped = len(df_clean) - len(df_final)
    print(f"Dropped {num_dropped} rows due to missing SMILES or invalid structures.")
    
    df_final.to_csv(output_path, index=False)
    print(f"Processing complete. Clean data saved to '{output_path}'.")
    print(f"Final dataset shape: {df_final.shape}")


if __name__ == '__main__':
    RAW_DATA_PATH = 'data/raw/DILIrank_dataset.csv'
    PROCESSED_DATA_PATH = 'data/processed/dili_data_clean.csv'
    
    process_raw_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)