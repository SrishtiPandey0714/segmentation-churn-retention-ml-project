import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the loaded telco churn dataset.
    - Convert 'Total Charges' to numeric
    - Drop rows where 'Total Charges' is null
    """
    df = df.copy()
    
    # Convert 'Total Charges' to numeric, setting errors to NaN
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    
    # Drop rows with NaN in 'Total Charges'
    df = df.dropna(subset=["Total Charges"])
    
    # Calculate ServiceCount as used later in features and segmentation
    service_cols = [
        'Phone Service', 'Multiple Lines', 'Internet Service',
        'Online Security', 'Online Backup', 'Device Protection',
        'Tech Support', 'Streaming TV', 'Streaming Movies'
    ]
    
    # Handle both string 'Yes' and boolean True
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1) + (df[service_cols] == True).sum(axis=1)
    
    return df
