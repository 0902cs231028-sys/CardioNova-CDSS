
# core/engine.py

import joblib
from catboost import CatBoostClassifier
from core.config import MODEL_PATH
from core.pipeline import load_artifacts, preprocess_data

def predict(data):
    """Make predictions using the pre-trained model."""
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    imputer, scaler = load_artifacts()
    processed_data = preprocess_data(data, imputer, scaler)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)
    return prediction, probability
