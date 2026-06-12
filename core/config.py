
# core/config.py

# Paths
MODEL_PATH = "artifacts/heart_failure_catboost_core.cbm"
IMPUTER_PATH = "artifacts/clinical_knn_imputer.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

# Features
NUMERIC_FEATURES = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"
]

CATEGORICAL_FEATURES = [
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"
]

TARGET_FEATURE = "HeartDisease"

# Clinical Rules Thresholds
HIGH_CHOLESTEROL_THRESHOLD = 240
HIGH_BP_THRESHOLD = 140
