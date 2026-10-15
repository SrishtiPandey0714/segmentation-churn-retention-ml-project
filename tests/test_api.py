import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from api.main import app, load_model
from src.features.build_features import build_features
from src.retention.strategy import get_risk_tier, recommend_retention_actions
from src.models.segmentation import CustomerSegmentation
from src.models.evaluate import evaluate_classification_model

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    load_model()

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "health_url" in data

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "version" in data

def test_predict_churn_endpoint():
    sample_customer = {
        "CustomerID": "test_1",
        "Gender": "Female",
        "Senior_Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 24,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "DSL",
        "Online Security": "Yes",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "Yes",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "One year",
        "Paperless Billing": "Yes",
        "Payment Method": "Mailed check",
        "Monthly Charges": 65.0,
        "Total Charges": 1560.0
    }
    
    response = client.post("/predict", json=sample_customer)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["customer_id"] == "test_1"
        assert 0.0 <= data["churn_prob"] <= 1.0
        assert data["churn_prediction"] in [0, 1]
        assert data["risk_level"] in ["Low", "Moderate", "Critical"]
        assert isinstance(data["retention_recommendations"], list)
        assert len(data["retention_recommendations"]) > 0

def test_predict_alias_support():
    # Pass with space format "Senior Citizen" instead of "Senior_Citizen"
    sample_customer = {
        "CustomerID": "test_alias",
        "Gender": "Male",
        "Senior Citizen": "Yes",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 3,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 95.0,
        "Total Charges": 285.0
    }
    
    response = client.post("/predict", json=sample_customer)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert 0.0 <= data["churn_prob"] <= 1.0
        assert data["risk_level"] in ["Low", "Moderate", "Critical"]

def test_explain_churn_endpoint():
    sample_customer = {
        "CustomerID": "test_explain",
        "Gender": "Male",
        "Senior_Citizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "Tenure Months": 12,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 89.50,
        "Total Charges": 1074.00
    }
    response = client.post("/explain", json=sample_customer)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert data["customer_id"] == "test_explain"
        assert isinstance(data["shap_values"], list)
        assert isinstance(data["feature_names"], list)
        assert len(data["shap_values"]) == len(data["feature_names"])

def test_retention_strategy_logic():
    assert get_risk_tier(0.85) == "Critical"
    assert get_risk_tier(0.55) == "Moderate"
    assert get_risk_tier(0.20) == "Low"

    high_risk_cust = {
        "Contract": "Month-to-month",
        "Monthly Charges": 90.0,
        "Internet Service": "Fiber optic",
        "Tenure Months": 4,
        "Payment Method": "Electronic check",
        "Tech Support": "No",
        "Online Security": "No"
    }
    recs = recommend_retention_actions(high_risk_cust, 0.78)
    assert len(recs) > 0
    assert any("annual" in r.lower() or "contract" in r.lower() for r in recs)

def test_feature_engineering():
    raw_dict = {
        "CustomerID": ["CUST-1"],
        "Tenure Months": [12],
        "Monthly Charges": [80.0],
        "Total Charges": [960.0],
        "Contract": ["Month-to-month"],
        "Internet Service": ["Fiber optic"],
        "Phone Service": ["Yes"],
        "Multiple Lines": ["No"],
        "Online Security": ["No"],
        "Online Backup": ["Yes"],
        "Device Protection": ["No"],
        "Tech Support": ["No"],
        "Streaming TV": ["Yes"],
        "Streaming Movies": ["No"]
    }
    df = pd.DataFrame(raw_dict)
    X, _ = build_features(df, is_train=False)
    assert "charges_per_month" in X.columns
    assert "avg_service_cost" in X.columns
    assert "log_total_charges" in X.columns
    assert X["charges_per_month"].iloc[0] > 0

def test_customer_segmentation():
    df = pd.DataFrame({
        "Tenure Months": [1, 24, 60, 12, 36],
        "Monthly Charges": [20.0, 70.0, 110.0, 50.0, 85.0],
        "Total Charges": [20.0, 1680.0, 6600.0, 600.0, 3060.0],
        "Phone Service": ["Yes", "Yes", "Yes", "Yes", "Yes"],
        "Multiple Lines": ["No", "Yes", "Yes", "No", "Yes"],
        "Internet Service": ["No", "DSL", "Fiber optic", "DSL", "Fiber optic"],
        "Online Security": ["No", "Yes", "Yes", "No", "No"],
        "Online Backup": ["No", "Yes", "Yes", "No", "Yes"],
        "Device Protection": ["No", "No", "Yes", "No", "Yes"],
        "Tech Support": ["No", "Yes", "Yes", "No", "No"],
        "Streaming TV": ["No", "No", "Yes", "No", "Yes"],
        "Streaming Movies": ["No", "No", "Yes", "No", "Yes"]
    })
    seg = CustomerSegmentation(n_clusters=2, random_state=42)
    seg.fit(df)
    preds = seg.predict(df)
    assert len(preds) == len(df)
    assert set(preds).issubset({0, 1})
