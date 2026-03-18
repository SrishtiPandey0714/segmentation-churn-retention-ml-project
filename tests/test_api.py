from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_churn_no_model_loaded():
    # If the model is not trained/loaded yet, it should return 503
    # We can mock MODEL but let's test the endpoint response
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
    # Could be 200 if model represents, or 503 if not loaded
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert "churn_prob" in response.json()
        assert "churn_prediction" in response.json()

def test_explain_churn_no_model_loaded():
    sample_customer = {
        "CustomerID": "test_2",
        "Tenure Months": 12,
        "Total Charges": 1000
    }
    response = client.post("/explain", json=sample_customer)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert "shap_values" in response.json()
