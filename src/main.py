import os
from dotenv import load_dotenv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import optuna
from sklearn.model_selection import train_test_split

from data_processing.balance_data import apply_smote, create_sample
from evaluation.compare import evaluate_model, plot_confusion, plot_feature_importance, plot_roc, save_metrics_table
from models.train_model import lgb_model_train, objective_lgb, objective_xgb, xgb_model_train
import models.model as model
from utils import utils

load_dotenv()
BASE_DIR = utils.get_base_dir()
INPUT_PATH = BASE_DIR / Path(os.getenv("INPUT_PROCESSED_PATH", "data/processed/")) / "credit_risk_cleaned.csv"

def main():
    """
    Train and evaluate credit risk prediction models using ensemble learning.
    
    Pipeline:
        1. Load and split data into train/validation/test sets (64/16/20 split)
        2. Apply SMOTE balancing on training set only to avoid data leakage
        3. Train five models: Baseline, two default configurations, two optimized
        4. Evaluate and compare performance on test set
    """
    
    # Load data and create train/test/validation splits
    df = utils.load_file(INPUT_PATH)
    X_train_full, X_test, y_train_full, y_test = create_sample(df)
    
    # Drop ID column to ensure all models use identical features
    if 'ID' in X_train_full.columns:
        X_train_full = X_train_full.drop('ID', axis=1)
        X_test = X_test.drop('ID', axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # Balance training data using SMOTE
    X_resample, y_resample = apply_smote(X_train=X_train, y_train=y_train)
    
    # For tuned models: apply SMOTE on full training set for retraining with best params
    X_resample_full, y_resample_full = apply_smote(X_train=X_train_full, y_train=y_train_full)
    
    # Initialize containers for results
    all_results = []
    all_probas = {}

    # Model 1: Baseline Logistic Regression (reference model)
    print("\n===== BASELINE LOGISTIC REGRESSION =====")
    baseline_pipeline = model.train_baseline_model(X_train, y_train)
    y_pred_base = baseline_pipeline.predict(X_test)
    y_proba_base = baseline_pipeline.predict_proba(X_test)[:, 1]
    metrics_base = evaluate_model(y_test, y_pred_base, y_proba_base)
    metrics_base["Model"] = "Logistic (Baseline)"
    all_results.append(metrics_base)
    all_probas["Logistic (Baseline)"] = y_proba_base

    # Model 2: LightGBM with default parameters
    print("\n===== LIGHTGBM (DEFAULT) =====")
    y_pred_lgb, y_proba_lgb, _ = lgb_model_train(
        X_resample=X_resample, y_resample=y_resample, X_test=X_test
    )
    metrics_lgb = evaluate_model(y_test=y_test, y_pred=y_pred_lgb, y_proba=y_proba_lgb)
    metrics_lgb["Model"] = "LightGBM (Default)"
    all_results.append(metrics_lgb)
    all_probas["LightGBM (Default)"] = y_proba_lgb
    
    # Model 3: XGBoost with default parameters
    print("\n===== XGBOOST (DEFAULT) =====")
    y_pred_xgb, y_proba_xgb, _ = xgb_model_train(
        X_resample=X_resample, y_resample=y_resample, X_test=X_test
    )
    metrics_xgb = evaluate_model(y_test=y_test, y_pred=y_pred_xgb, y_proba=y_proba_xgb)
    metrics_xgb["Model"] = "XGBoost (Default)"
    all_results.append(metrics_xgb)
    all_probas["XGBoost (Default)"] = y_proba_xgb
    
    # Model 4: XGBoost with Bayesian hyperparameter optimization (30 trials)
    print("\n===== OPTUNA TUNING XGBOOST =====")
    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(
        lambda trial: objective_xgb(
            trial=trial, X_resample=X_resample, y_resample=y_resample, 
            X_test=X_val, y_test=y_val
        ), 
        n_trials=30
    )
    y_pred_xgb_best, y_proba_xgb_best, xgb_best_model = xgb_model_train(
        X_resample=X_resample_full, y_resample=y_resample_full, X_test=X_test, 
        **study_xgb.best_params
    )
    metrics_xgb_best = evaluate_model(y_test, y_pred_xgb_best, y_proba_xgb_best)
    metrics_xgb_best["Model"] = "XGBoost (Tuned)"
    all_results.append(metrics_xgb_best)
    all_probas["XGBoost (Tuned)"] = y_proba_xgb_best

    # Model 5: LightGBM with Bayesian hyperparameter optimization (30 trials)
    print("\n===== OPTUNA TUNING LIGHTGBM =====")
    study_lgb = optuna.create_study(direction="maximize")
    study_lgb.optimize(
        lambda trial: objective_lgb(
            trial=trial, X_resample=X_resample, y_resample=y_resample, 
            X_test=X_val, y_test=y_val
        ), 
        n_trials=30
    )
    y_pred_lgb_best, y_proba_lgb_best, lgb_best_model = lgb_model_train(
        X_resample=X_resample_full, y_resample=y_resample_full, X_test=X_test, 
        **study_lgb.best_params
    )
    metrics_lgb_best = evaluate_model(y_test, y_pred_lgb_best, y_proba_lgb_best)
    metrics_lgb_best["Model"] = "LightGBM (Tuned)"
    all_results.append(metrics_lgb_best)
    all_probas["LightGBM (Tuned)"] = y_proba_lgb_best

    # Evaluation and comparison of all models
    print("\n" + "="*70)
    print("MODEL EVALUATION & COMPARISON")
    print("="*70)
    
    # Generate and save comparison metrics
    save_metrics_table(all_results, "final_model_comparison.csv")
    
    # Generate ROC curve comparison for all models
    plot_roc(y_test, all_probas, filename="final_roc_comparison")
    
    # Generate confusion matrices for best performing models
    plot_confusion(y_test, y_pred_xgb_best, "confusion_xgb_tuned.png", "XGBoost (Tuned)")
    plot_confusion(y_test, y_pred_lgb_best, "confusion_lgb_tuned.png", "LightGBM (Tuned)")
    
    # Generate feature importance plot from best model (LightGBM Tuned)
    plot_feature_importance(lgb_best_model, X_train, "feature_importance_best.png")

    from evaluation.compare import OUTPUT_PATH
    print("\n" + "="*70)
    print(f"Evaluation complete. Results saved to: {OUTPUT_PATH}")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
