from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import os
import joblib
import shap
import json
from fastapi.middleware.cors import CORSMiddleware
import shap
import json

from api.schemas import CustomerInput, PredictResponse
from src.features.build_features import build_features

app = FastAPI(
    title="Customer Retention Intelligence Platform (CRIP)",
    description="API for predicting customer churn and explaining predictions.",
    version="1.0"
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
MODEL = None

@app.on_event("startup")
def load_model():
    global MODEL
    model_path = os.getenv("MODEL_PATH", "src/models/churn_model.pkl")
    if os.path.exists(model_path):
        MODEL = joblib.load(model_path)
    else:
        print(f"Warning: Model not found at {model_path}. Please train the model first.")

@app.get("/")
def read_root():
    return {"message": "Welcome to CRIP API. Go to /docs for the Swagger UI."}

@app.post("/predict", response_model=PredictResponse)
def predict_churn(customer: CustomerInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    # Convert input to DataFrame
    customer_dict = customer.dict(by_alias=True)
    df = pd.DataFrame([customer_dict])
    
    # Prepare features
    X, _ = build_features(df, is_train=False)
    
    # Align columns to what the model expects
    if hasattr(MODEL, "lr") and hasattr(MODEL.lr, "feature_names_in_"):
        expected_cols = MODEL.lr.feature_names_in_
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]
    
    # Predict
    try:
        prob = float(MODEL.predict_proba(X)[0, 1])
        prediction = int(MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    return PredictResponse(
        customer_id=customer.CustomerID,
        churn_prob=prob,
        churn_prediction=prediction
    )

@app.post("/explain")
def explain_churn(customer: CustomerInput):
    """
    Returns SHAP values explaining the model's prediction for this customer.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    # Convert input to DataFrame
    customer_dict = customer.dict(by_alias=True)
    df = pd.DataFrame([customer_dict])
    
    # Prepare features
    X, _ = build_features(df, is_train=False)
    
    # Align columns to what the model expects
    if hasattr(MODEL, "lr") and hasattr(MODEL.lr, "feature_names_in_"):
        expected_cols = MODEL.lr.feature_names_in_
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_cols]
    
    try:
        # We need the underlying LightGBM model for TreeExplainer
        # Assuming our ensemble has `lgbm` property
        explainer = shap.TreeExplainer(MODEL.lgbm)
        shap_values = explainer.shap_values(X)
        
        # shap_values could be a list for multi-class, for binary we usually take [1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0].tolist()
        else:
            sv = shap_values[0].tolist()
            
        return {
            "customer_id": customer.CustomerID,
            "shap_values": sv,
            "feature_names": X.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
