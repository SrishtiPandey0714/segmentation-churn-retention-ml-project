import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, is_train: bool = True):
    """
    Convert raw data into features ready for model training or inference.
    """
    df = df.copy()

    # Normalize aliases if present
    if "Senior_Citizen" in df.columns and "Senior Citizen" not in df.columns:
        df["Senior Citizen"] = df["Senior_Citizen"]
    if "Tenure_Months" in df.columns and "Tenure Months" not in df.columns:
        df["Tenure Months"] = df["Tenure_Months"]
    if "Monthly_Charges" in df.columns and "Monthly Charges" not in df.columns:
        df["Monthly Charges"] = df["Monthly_Charges"]
    if "Total_Charges" in df.columns and "Total Charges" not in df.columns:
        df["Total Charges"] = df["Total_Charges"]

    # Calculate ServiceCount if missing
    if "ServiceCount" not in df.columns:
        service_cols = [
            'Phone Service', 'Multiple Lines', 'Internet Service',
            'Online Security', 'Online Backup', 'Device Protection',
            'Tech Support', 'Streaming TV', 'Streaming Movies'
        ]
        existing = [c for c in service_cols if c in df.columns]
        if existing:
            df["ServiceCount"] = (df[existing] == "Yes").sum(axis=1) + (df[existing] == True).sum(axis=1)
        else:
            df["ServiceCount"] = 0

    # Ensure numeric columns are floats
    for col in ["Total Charges", "Monthly Charges", "Tenure Months", "ServiceCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Define target if it's training data
    y = None
    if "Churn Value" in df.columns:
        y = df["Churn Value"]

    # Drop unnecessary columns
    drop_cols = [
        "CustomerID", "Churn Label", "Churn Score", 
        "Churn Reason", "CLTV", "Churn Value",
        "Count", "Country", "State", "City", 
        "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Segment", "Senior_Citizen"
    ]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Feature Engineering
    tenure = X["Tenure Months"] if "Tenure Months" in X.columns else 0
    total_charges = X["Total Charges"] if "Total Charges" in X.columns else 0
    monthly_charges = X["Monthly Charges"] if "Monthly Charges" in X.columns else 0
    service_count = X["ServiceCount"] if "ServiceCount" in X.columns else 0

    X["charges_per_month"] = total_charges / (tenure + 1)
    X["avg_service_cost"] = monthly_charges / (service_count + 1)
    
    if "Contract_One year" in X.columns:
        X["tenure_contract_1yr"] = tenure * X["Contract_One year"]
    else:
        X["tenure_contract_1yr"] = 0
        
    if "Contract_Two year" in X.columns:
        X["tenure_contract_2yr"] = tenure * X["Contract_Two year"]
    else:
        X["tenure_contract_2yr"] = 0
        
    if "Internet Service_Fiber optic" in X.columns:
        X["fiber_charges"] = monthly_charges * X["Internet Service_Fiber optic"]
        is_fiber = X["Internet Service_Fiber optic"] == 1
    else:
        X["fiber_charges"] = 0
        is_fiber = False

    service_cols = [
        "Online Security_Yes", "Online Backup_Yes",
        "Device Protection_Yes", "Tech Support_Yes",
        "Streaming TV_Yes", "Streaming Movies_Yes"
    ]
    
    existing_service_cols = [col for col in service_cols if col in X.columns]
    X["total_active_services"] = X[existing_service_cols].sum(axis=1) if existing_service_cols else 0
    
    median_monthly_charge = monthly_charges.median() if hasattr(monthly_charges, "median") else 0
    X["fiber_high_pay"] = (is_fiber & (monthly_charges > median_monthly_charge)).astype(int)
    
    X["log_total_charges"] = np.log1p(total_charges)
    
    return X, y
