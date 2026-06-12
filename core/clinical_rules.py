
# core/clinical_rules.py

from core.config import HIGH_CHOLESTEROL_THRESHOLD, HIGH_BP_THRESHOLD

def check_clinical_rules(data):
    """Check for clinical rule violations."""
    alerts = []
    if data.get("Cholesterol", 0) > HIGH_CHOLESTEROL_THRESHOLD:
        alerts.append("High cholesterol detected.")
    if data.get("RestingBP", 0) > HIGH_BP_THRESHOLD:
        alerts.append("High blood pressure detected.")
    return alerts
