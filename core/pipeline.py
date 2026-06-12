
# core/pipeline.py

import joblib
import pandas as pd
from core.config import IMPUTER_PATH, SCALER_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def load_artifacts():
    """Load the pre-trained imputer and scaler."""
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return imputer, scaler

def preprocess_data(data, imputer, scaler):
    """Preprocess the input data using the loaded artifacts."""
    df = pd.DataFrame(data, index=[0])
    df[NUMERIC_FEATURES] = imputer.transform(df[NUMERIC_FEATURES])
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES])
    return df
