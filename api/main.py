from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import uvicorn
import os
import joblib
import shap
import json
import logging

from api.schemas import CustomerInput, PredictResponse, ExplainResponse, HealthResponse
from src.features.build_features import build_features
from src.retention.strategy import get_risk_tier, recommend_retention_actions
from src.utils.logger import get_logger

logger = get_logger("crip-api")

# Global model variable
MODEL = None

def load_model():
    global MODEL
    model_path = os.getenv("MODEL_PATH", "src/models/churn_model.pkl")
    if os.path.exists(model_path):
        try:
            MODEL = joblib.load(model_path)
            logger.info(f"Model successfully loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            MODEL = None
    else:
        logger.warning(f"Model not found at {model_path}. Prediction endpoints will return 503 until model is trained.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down CRIP API.")

app = FastAPI(
    title="Customer Retention Intelligence Platform (CRIP)",
    description="Production API for customer segmentation, churn prediction, SHAP explanations, and retention recommendations.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Customer Retention Intelligence Platform (CRIP) API.",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=(MODEL is not None),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse)
def predict_churn(customer: CustomerInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train or provide model.")
    
    # Convert input to DataFrame using alias mapping
    customer_dict = customer.model_dump(by_alias=True) if hasattr(customer, "model_dump") else customer.dict(by_alias=True)
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
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    risk_level = get_risk_tier(prob)
    recommendations = recommend_retention_actions(customer_dict, prob)

    return PredictResponse(
        customer_id=customer.CustomerID,
        churn_prob=prob,
        churn_prediction=prediction,
        risk_level=risk_level,
        retention_recommendations=recommendations
    )

@app.post("/explain")
def explain_churn(customer: CustomerInput):
    """
    Returns SHAP values explaining the model's prediction for this customer.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train or provide model.")
    
    # Convert input to DataFrame using alias mapping
    customer_dict = customer.model_dump(by_alias=True) if hasattr(customer, "model_dump") else customer.dict(by_alias=True)
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
        explainer = shap.TreeExplainer(MODEL.lgbm)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0].tolist() if len(shap_values) > 1 else shap_values[0][0].tolist()
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            sv = shap_values[0, :, 1].tolist()
        elif hasattr(shap_values, "tolist"):
            sv = shap_values[0].tolist()
        else:
            sv = list(shap_values[0])
            
        return {
            "customer_id": customer.CustomerID,
            "shap_values": sv,
            "feature_names": X.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
