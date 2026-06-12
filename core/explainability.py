
# core/explainability.py

import shap
import pandas as pd
from catboost import CatBoostClassifier
from core.config import MODEL_PATH, NUMERIC_FEATURES
from core.pipeline import preprocess_data, load_artifacts

def get_shap_explainer():
    """Initialize the SHAP explainer."""
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    background_df = pd.DataFrame(columns=NUMERIC_FEATURES)
    explainer = shap.TreeExplainer(model, background_df)
    return explainer

def get_shap_values(explainer, data):
    """Get SHAP values for the input data."""
    imputer, scaler = load_artifacts()
    processed_data = preprocess_data(data, imputer, scaler)
    shap_values = explainer.shap_values(processed_data)
    return shap_values
