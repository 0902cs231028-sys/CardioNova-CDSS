
import pandas as pd
import numpy as np
import joblib
import json
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
import warnings
from pathlib import Path
import sys

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)

# --- Constants ---
# NOTE: Ensure this path points to your dataset.
# The Z-Alizadeh Sani dataset can be found on Kaggle or the UCI Machine Learning Repository.
DATA_PATH = Path("z-clean-z-alizadeh-sani-dataset.csv") 
ARTIFACTS_DIR = Path("artifacts")
TARGET = "Cath"
SEED = 42

def main():
    """Main function to run the training pipeline."""
    
    # --- 1. Load Data ---
    print("🚀 Starting training pipeline...")
    try:
        df = pd.read_csv(DATA_PATH)
        print("✅ Data loaded successfully.")
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at '{DATA_PATH}'.")
        print("Please download the 'Z-Alizadeh Sani' dataset and place it in the root directory.")
        sys.exit(1)

    # --- 2. Preprocessing ---
    print("\n🔄 Preprocessing and feature engineering...")
    
    # For consistency with the Streamlit app's feature names
    df.rename(columns={"Cath": "HeartDisease"}, inplace=True)
    global TARGET
    TARGET = "HeartDisease"

    # Define feature types for CatBoost
    categorical_cols = [
        'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope',
        'DM', 'HTN', 'Current Smoker', 'EX-Smoker', 'FH', 'Obesity', 'CRF',
        'CVA', 'Airway disease', 'Thyroid Disease', 'CHF', 'DLP', 'Edema',
        'Weak Peripheral Pulse', 'Lung rales', 'Systolic Murmur', 'Diastolic Murmur',
        'Typical Chest Pain', 'Dyspnea', 'Atypical', 'Nonanginal', 'Exertional CP',
        'LowTH Ang', 'Q Wave', 'St Elevation', 'St Depression', 'Tinversion', 'LVH',
        'Poor R Progression', 'Function Class', 'Region RWMA', 'VHD'
    ]
    
    # Filter out columns not present in the dataframe
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    y = df[TARGET]
    X = df.drop(TARGET, axis=1)
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    
    # --- 3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"Split data: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- 4. Fit Imputer & Scaler ---
    print("\n🛠️ Fitting preprocessing artifacts (Imputer, Scaler)...")
    imputer = KNNImputer(n_neighbors=5)
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # --- 5. Handle Class Imbalance ---
    print("\n⚖️ Balancing classes using SMOTE...")
    smote = SMOTE(random_state=SEED)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # --- 6. Hyperparameter Tuning (Optuna) ---
    print("\n🧠 Optimizing hyperparameters with Optuna...")
    cat_indices = [X.columns.get_loc(col) for col in categorical_cols if col in X.columns]

    def objective(trial):
        params = {
            'objective': 'Logloss',
            'eval_metric': 'Recall',
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'verbose': 0, 'random_seed': SEED
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train_balanced, y_train_balanced, cat_features=cat_indices, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_test)
        return recall_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    print(f"🏆 Best trial recall: {study.best_value:.4f}")
    print(f"📋 Best params: {study.best_params}")

    # --- 7. Train Final Model ---
    print("\n💪 Training final model on all data with best parameters...")
    final_model = CatBoostClassifier(**study.best_params, random_seed=SEED, verbose=False)
    
    # Combine original training and test sets, then balance all data for final training
    X_full_imputed_scaled = pd.concat([pd.DataFrame(X_train, columns=X.columns), pd.DataFrame(X_test, columns=X.columns)])
    y_full = pd.concat([y_train, y_test])
    X_final, y_final = smote.fit_resample(X_full_imputed_scaled, y_full)
    
    final_model.fit(X_final, y_final, cat_features=cat_indices)
    
    # --- 8. Save Artifacts ---
    print("\n💾 Saving all artifacts...")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    
    # Model
    model_path = ARTIFACTS_DIR / "heart_failure_catboost_core.cbm"
    final_model.save_model(model_path)
    
    # Imputer
    imputer_path = ARTIFACTS_DIR / "clinical_knn_imputer.pkl"
    joblib.dump(imputer, imputer_path)
    
    # Scaler
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Feature Schema
    schema = {
        "numeric_features": numeric_cols,
        "categorical_features": [col for col in X.columns if col in categorical_cols],
        "strict_feature_order": X.columns.tolist(),
        "target_variable": TARGET
    }
    schema_path = ARTIFACTS_DIR / "feature_schema_lock.json"
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=4)
        
    print(f"  -> Model saved: {model_path}")
    print(f"  -> Imputer saved: {imputer_path}")
    print(f"  -> Scaler saved: {scaler_path}")
    print(f"  -> Schema saved: {schema_path}")
    
    print("\n🎉 Training pipeline finished successfully!")


if __name__ == "__main__":
    main()
