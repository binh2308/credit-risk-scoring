import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import joblib

from utils.utils import get_base_dir

# Define paths
BASE_DIR = get_base_dir()
DATA_PATH = BASE_DIR / ("data/processed/credit_risk_cleaned.csv")
MODEL_DIR = BASE_DIR / Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Define columns
TARGET = "default_payment_next_month"

# Categorical variables for one-hot encoding
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Monetary variables for standard scaling
MONETARY_COLS = [
    "LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# Other numerical columns (not scaled)
OTHER_NUM_COLS = ["AGE", "credit_util_ratio", "payment_ratio", "late_payment_count"]

def load_data():
    """Load the cleaned dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded data shape: {df.shape}")
    return df

def split_data(df, test_size=0.2, random_state=42):
    """Split data into train and test sets (80% train, 20% test)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train target distribution: {y_train.value_counts(normalize=True)}")
    print(f"Test target distribution: {y_test.value_counts(normalize=True)}")

    return X_train, X_test, y_train, y_test

def create_preprocessor():
    """Create a ColumnTransformer for preprocessing: One-Hot for categorical, Standard Scale for monetary."""
    # One-hot encoder for categorical variables
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    # Standard scaler for monetary variables
    monetary_transformer = StandardScaler()

    # Pass through for other numerical columns
    other_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, CATEGORICAL_COLS),
            ('mon', monetary_transformer, MONETARY_COLS),
            ('num', other_transformer, OTHER_NUM_COLS)
        ],
        remainder='drop'  # Drop any columns not specified
    )

    return preprocessor

def train_baseline_model(X_train, y_train):
    """Train Logistic Regression baseline model on unbalanced train set."""
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)
    print("Baseline Logistic Regression model trained successfully.")

    return pipeline

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return ROC-AUC score."""
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Baseline ROC-AUC: {roc_auc:.4f}")

    # Classification report
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return roc_auc

def save_model(model, filename="baseline_logistic_regression.pkl"):
    """Save the trained model."""
    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

def main():
    # Load data
    df = load_data()

    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_data(df)

    # Train baseline model (Logistic Regression on unbalanced train)
    model = train_baseline_model(X_train, y_train)

    # Evaluate model (get base ROC-AUC)
    roc_auc = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

    print("\nBaseline model training completed!")
    print(f"Final ROC-AUC on test set: {roc_auc:.4f}")

if __name__ == "__main__":
    main()