import urllib.request
import urllib.error
import json

sample_customer = {
    "CustomerID": "CUST-001",
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

data = json.dumps(sample_customer).encode("utf-8")
req = urllib.request.Request("http://localhost:8000/explain", data=data, headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req) as response:
        print("Success:", response.read().decode())
except urllib.error.HTTPError as e:
    with open("error.txt", "w") as f:
        f.write(e.read().decode())
    print("Error written to error.txt")
except Exception as e:
    print("Other Error:", str(e))
