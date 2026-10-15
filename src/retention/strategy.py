from typing import Dict, Any, List

def get_risk_tier(churn_prob: float) -> str:
    """
    Classify customer into a risk tier based on churn probability.
    """
    if churn_prob >= 0.70:
        return "Critical"
    elif churn_prob >= 0.40:
        return "Moderate"
    else:
        return "Low"

def recommend_retention_actions(customer_data: Dict[str, Any], churn_prob: float) -> List[str]:
    """
    Generate proactive retention recommendations based on customer attributes and risk level.
    """
    actions = []
    risk = get_risk_tier(churn_prob)
    
    if risk == "Low":
        actions.append("Customer is healthy. Enroll in routine loyalty reward programs.")
        return actions

    # Month-to-month contracts are primary churn drivers
    contract = customer_data.get("Contract", "")
    if contract == "Month-to-month":
        actions.append("Offer 15% discount for switching to an annual (1-year) contract.")

    # High charges or fiber optic concerns
    monthly_charges = float(customer_data.get("Monthly Charges", customer_data.get("Monthly_Charges", 0.0)))
    internet_service = customer_data.get("Internet Service", customer_data.get("Internet_Service", ""))
    if monthly_charges > 70.0 and internet_service == "Fiber optic":
        actions.append("Review fiber optic bill: Offer loyalty bundle credit or free speed bump.")

    # Lack of security/tech support services
    tech_support = customer_data.get("Tech Support", customer_data.get("Tech_Support", "No"))
    online_security = customer_data.get("Online Security", customer_data.get("Online_Security", "No"))
    if tech_support == "No" or online_security == "No":
        actions.append("Provide 3 months complimentary Tech Support & Online Security pack.")

    # New vs Tenured customer strategies
    tenure = int(customer_data.get("Tenure Months", customer_data.get("Tenure_Months", 0)))
    if tenure <= 6:
        actions.append("Schedule proactive customer success onboarding check-in call.")
    elif tenure > 24 and risk in ["Critical", "Moderate"]:
        actions.append("Assign Senior Account Manager for personalized VIP retention outreach.")

    # Payment method friction
    payment_method = customer_data.get("Payment Method", customer_data.get("Payment_Method", ""))
    if "check" in payment_method.lower():
        actions.append("Incentivize enrollment in automatic bank/credit card payment with a $10 one-time credit.")

    if not actions:
        actions.append("Provide standard proactive customer service survey and loyalty discount.")

    return actions
