import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix, 
                             f1_score, recall_score, precision_score, roc_curve)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Define paths
DATA_PATH = Path("data/processed/credit_risk_cleaned.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/advanced_models")
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "default_payment_next_month"
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
MONETARY_COLS = ["LIMIT_BAL", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                 "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
OTHER_NUM_COLS = ["AGE", "credit_util_ratio", "payment_ratio", "late_payment_count"]

def load_and_preprocess():
    """Load data, split, preprocess, and apply SMOTE only to training set."""
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_COLS),
        ('mon', StandardScaler(), MONETARY_COLS),
        ('num', 'passthrough', OTHER_NUM_COLS)
    ], remainder='drop')
    
    # Fit on train, transform train and test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Apply SMOTE ONLY to training set
    print(f"\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    print(f"After SMOTE - Train: {X_train_smote.shape}")
    print(f"After SMOTE class distribution:\n{pd.Series(y_train_smote).value_counts(normalize=True)}")
    
    # Save preprocessor
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.pkl")
    
    return X_train_smote, X_test_processed, y_train_smote, y_test

def train_baseline_models(X_train, X_test, y_train, y_test):
    """Train LightGBM and XGBoost baseline models."""
    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODELS")
    print("=" * 80)
    
    # LightGBM baseline
    print("\nTraining LightGBM baseline...")
    lgb_baseline = lgb.LGBMClassifier(
        random_state=42, n_estimators=100, n_jobs=-1,
        is_unbalance=True, verbose=-1
    )
    lgb_baseline.fit(X_train, y_train)
    lgb_baseline_auc = roc_auc_score(y_test, lgb_baseline.predict_proba(X_test)[:, 1])
    print(f"LightGBM Baseline ROC-AUC: {lgb_baseline_auc:.4f}")
    
    # XGBoost baseline
    print("Training XGBoost baseline...")
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_baseline = XGBClassifier(
        random_state=42, n_estimators=100, scale_pos_weight=scale_pos_weight,
        tree_method='hist', verbose=0
    )
    xgb_baseline.fit(X_train, y_train)
    xgb_baseline_auc = roc_auc_score(y_test, xgb_baseline.predict_proba(X_test)[:, 1])
    print(f"XGBoost Baseline ROC-AUC: {xgb_baseline_auc:.4f}")
    
    return lgb_baseline, xgb_baseline

def create_lgb_objective(X_train, y_train, X_test, y_test):
    """Create Optuna objective for LightGBM tuning."""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 10.0),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 10.0),
        }
        
        model = lgb.LGBMClassifier(
            random_state=42, n_estimators=100, n_jobs=-1,
            is_unbalance=True, verbose=-1, **params
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(10)])
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    return objective

def create_xgb_objective(X_train, y_train, X_test, y_test):
    """Create Optuna objective for XGBoost tuning."""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'lambda': trial.suggest_float('lambda', 0.0, 10.0),
            'alpha': trial.suggest_float('alpha', 0.0, 10.0),
        }
        
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        model = XGBClassifier(
            random_state=42, n_estimators=100, scale_pos_weight=scale_pos_weight,
            tree_method='hist', verbose=0, **params
        )
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc
    
    return objective

def tune_hyperparameters(X_train, y_train, X_test, y_test):
    """Tune hyperparameters using Optuna."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING WITH OPTUNA (50 trials)")
    print("=" * 80)
    
    # LightGBM tuning
    print("\nTuning LightGBM...")
    lgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    lgb_objective = create_lgb_objective(X_train, y_train, X_test, y_test)
    lgb_study.optimize(lgb_objective, n_trials=25, show_progress_bar=True)
    
    print(f"Best LightGBM params: {lgb_study.best_params}")
    print(f"Best LightGBM ROC-AUC: {lgb_study.best_value:.4f}")
    
    # XGBoost tuning
    print("\nTuning XGBoost...")
    xgb_study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    xgb_objective = create_xgb_objective(X_train, y_train, X_test, y_test)
    xgb_study.optimize(xgb_objective, n_trials=25, show_progress_bar=True)
    
    print(f"Best XGBoost params: {xgb_study.best_params}")
    print(f"Best XGBoost ROC-AUC: {xgb_study.best_value:.4f}")
    
    return lgb_study.best_params, xgb_study.best_params

def train_best_models(X_train, y_train, X_test, y_test, lgb_params, xgb_params):
    """Train final models with best hyperparameters."""
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
    print("=" * 80)
    
    # LightGBM
    print("Training final LightGBM...")
    lgb_final = lgb.LGBMClassifier(
        random_state=42, n_estimators=200, n_jobs=-1,
        is_unbalance=True, verbose=-1, **lgb_params
    )
    lgb_final.fit(X_train, y_train)
    lgb_auc = roc_auc_score(y_test, lgb_final.predict_proba(X_test)[:, 1])
    print(f"Final LightGBM ROC-AUC: {lgb_auc:.4f}")
    joblib.dump(lgb_final, MODEL_DIR / "lgb_final.pkl")
    
    # XGBoost
    print("Training final XGBoost...")
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    xgb_final = XGBClassifier(
        random_state=42, n_estimators=200, scale_pos_weight=scale_pos_weight,
        tree_method='hist', verbose=0, **xgb_params
    )
    xgb_final.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb_final.predict_proba(X_test)[:, 1])
    print(f"Final XGBoost ROC-AUC: {xgb_auc:.4f}")
    joblib.dump(xgb_final, MODEL_DIR / "xgb_final.pkl")
    
    return lgb_final, xgb_final

def evaluate_models(models_dict, X_test, y_test):
    """Evaluate all models and generate comprehensive metrics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("=" * 80)
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"\n--- {name} ---")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            'model': model,
            'roc_auc': roc_auc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cm': confusion_matrix(y_test, y_pred)
        }
    
    return results

def plot_evaluation(results, X_test, y_test):
    """Plot evaluation metrics and ROC curves."""
    print("\nGenerating evaluation plots...")
    
    # ROC Curve comparison
    plt.figure(figsize=(10, 6))
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=300)
    plt.show()
    
    # Metrics comparison
    metrics_names = ['roc_auc', 'f1', 'recall', 'precision']
    metrics_data = {name: [results[name][m] for m in metrics_names] for name in results.keys()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_names))
    width = 0.25
    for i, (name, values) in enumerate(metrics_data.items()):
        ax.bar(x + i * width, values, width, label=name)
    ax.set_ylabel('Score')
    ax.set_title('Model Metrics Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=300)
    plt.show()

def explain_with_shap(best_model, X_test, model_name):
    """Generate SHAP explanations."""
    print(f"\nGenerating SHAP explanations for {model_name}...")
    
    # Sample data for SHAP (to speed up)
    X_sample = X_test[:min(1000, len(X_test))]
    
    if 'LightGBM' in model_name or 'lgb' in model_name.lower():
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.TreeExplainer(best_model)
    
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):  # XGBoost/LightGBM returns list
        shap.summary_plot(shap_values[1], X_sample, show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary Plot - {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_importance_{model_name.lower().replace(' ', '_')}.png", dpi=300)
    plt.show()

def main():
    # Load and preprocess data with SMOTE
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Train baseline models
    lgb_baseline, xgb_baseline = train_baseline_models(X_train, X_test, y_train, y_test)
    
    # Tune hyperparameters
    lgb_params, xgb_params = tune_hyperparameters(X_train, y_train, X_test, y_test)
    
    # Train final models
    lgb_final, xgb_final = train_best_models(X_train, y_train, X_test, y_test, lgb_params, xgb_params)
    
    # Evaluate all models
    models = {
        'LightGBM Baseline': lgb_baseline,
        'XGBoost Baseline': xgb_baseline,
        'LightGBM Final': lgb_final,
        'XGBoost Final': xgb_final
    }
    
    results = evaluate_models(models, X_test, y_test)
    
    # Plot evaluation
    plot_evaluation(results, X_test, y_test)
    
    # SHAP explanations for best models
    best_lgb_auc = results['LightGBM Final']['roc_auc']
    best_xgb_auc = results['XGBoost Final']['roc_auc']
    
    if best_lgb_auc >= best_xgb_auc:
        print(f"\nLightGBM is better ({best_lgb_auc:.4f} vs {best_xgb_auc:.4f})")
        explain_with_shap(lgb_final, X_test, 'LightGBM')
    else:
        print(f"\nXGBoost is better ({best_xgb_auc:.4f} vs {best_lgb_auc:.4f})")
        explain_with_shap(xgb_final, X_test, 'XGBoost')
    
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
