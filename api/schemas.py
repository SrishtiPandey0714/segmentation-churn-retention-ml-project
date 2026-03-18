from pydantic import BaseModel, Field
from typing import Optional

class CustomerInput(BaseModel):
    CustomerID: str = "customer_123"
    Gender: str = "Male"
    Senior_Citizen: str = "No"
    Partner: str = "No"
    Dependents: str = "No"
    Tenure_Months: int = Field(alias="Tenure Months", default=12)
    Phone_Service: str = Field(alias="Phone Service", default="Yes")
    Multiple_Lines: str = Field(alias="Multiple Lines", default="No")
    Internet_Service: str = Field(alias="Internet Service", default="Fiber optic")
    Online_Security: str = Field(alias="Online Security", default="No")
    Online_Backup: str = Field(alias="Online Backup", default="No")
    Device_Protection: str = Field(alias="Device Protection", default="No")
    Tech_Support: str = Field(alias="Tech Support", default="No")
    Streaming_TV: str = Field(alias="Streaming TV", default="Yes")
    Streaming_Movies: str = Field(alias="Streaming Movies", default="Yes")
    Contract: str = "Month-to-month"
    Paperless_Billing: str = Field(alias="Paperless Billing", default="Yes")
    Payment_Method: str = Field(alias="Payment Method", default="Electronic check")
    Monthly_Charges: float = Field(alias="Monthly Charges", default=89.50)
    Total_Charges: float = Field(alias="Total Charges", default=1074.00)

class PredictResponse(BaseModel):
    customer_id: str
    churn_prob: float
    churn_prediction: int
