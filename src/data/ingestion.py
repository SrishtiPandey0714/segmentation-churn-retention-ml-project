import pandas as pd
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the Telco Churn dataset from a CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df
