import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame, is_train: bool = True):
    """
    Convert raw data into features ready for model training or inference.
    """
    df = df.copy()
    
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
        "Segment" # Segmentation column added in notebook
    ]
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # Feature Engineering from Phase 9 of Notebook
    X["charges_per_month"] = X["Total Charges"] / (X["Tenure Months"] + 1)
    X["avg_service_cost"] = X["Monthly Charges"] / (X.get("ServiceCount", 0) + 1)
    
    # Check if these dummy variables exist before creating interaction features
    if "Contract_One year" in X.columns:
        X["tenure_contract_1yr"] = X["Tenure Months"] * X["Contract_One year"]
    else:
        X["tenure_contract_1yr"] = 0
        
    if "Contract_Two year" in X.columns:
        X["tenure_contract_2yr"] = X["Tenure Months"] * X["Contract_Two year"]
    else:
        X["tenure_contract_2yr"] = 0
        
    if "Internet Service_Fiber optic" in X.columns:
        X["fiber_charges"] = X["Monthly Charges"] * X["Internet Service_Fiber optic"]
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
    
    median_monthly_charge = X["Monthly Charges"].median() if not X.empty else 0
    X["fiber_high_pay"] = (is_fiber & (X["Monthly Charges"] > median_monthly_charge)).astype(int)
    
    X["log_total_charges"] = np.log1p(X["Total Charges"])
    
    return X, y
