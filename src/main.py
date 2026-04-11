import os
from dotenv import load_dotenv
from pathlib import Path

import optuna
from sklearn.model_selection import train_test_split

from data_processing.balance_data import apply_smote, create_sample
from evaluation.compare import evaluate_model, plot_confusion, plot_feature_importance, plot_roc
from models.train_model import lgb_model_train, objective_lgb, objective_xgb, xgb_model_train
import models.model as model
from utils import utils

load_dotenv()
BASE_DIR = utils.get_base_dir()
INPUT_PATH =BASE_DIR / Path(os.getenv("INPUT_PROCESSED_PATH", "data/processed/")) / "credit_risk_cleaned.csv"
    
def main():
  df = utils.load_file(INPUT_PATH)
  X_train_full, X_test, y_train_full, y_test  = create_sample(df)
  X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)
  
  
  X_resample, y_resample = apply_smote(X_train=X_train, y_train=y_train)
  print("\n===== TARGET VALUE COUNTS =====")
  print(y_resample.value_counts())

  print("\n===== LIGHTGBM MODEL =====")
  y_pred_lgb, y_proba_lgb = lgb_model_train(X_resample=X_resample, y_resample=y_resample, X_test=X_test)
  print(evaluate_model(y_test=y_test, y_pred=y_pred_lgb, y_proba=y_proba_lgb))
    
  print("\n===== XGBoost MODEL =====")
  y_pred_xgb, y_proba_xgb, xgb_model = xgb_model_train(X_resample=X_resample, y_resample=y_resample, X_test=X_test)
  print(evaluate_model(y_test=y_test, y_pred=y_pred_xgb, y_proba=y_proba_xgb))
  
  print("\n===== ROC =====")
  plot_roc(y_test=y_test, y_proba_lgb=y_proba_lgb, y_proba_xgb=y_proba_xgb)
  
  # Confusion Matrix
  print("\n===== CONFUSION MATRIX =====")
  plot_confusion(y_test, y_pred_lgb, "lightgbm", "LightGBM")
  plot_confusion(y_test, y_pred_xgb, "xgboost" ,"XGBoost")
  
  # Feature importance
  print("\n===== FEATURE IMPORTANCE =====")
  plot_feature_importance(xgb_model, X_train) 
  
  print("\n===== OPTUNA TUNING XGBOOST =====")
  study = optuna.create_study(direction="maximize")
  study.optimize(lambda trial: objective_xgb(trial=trial, X_resample=X_train, y_resample=y_train, X_test=X_val, y_test=y_val), n_trials=50)
  print("Best params:", study.best_params)
  print("Best validation ROC-AUC:", study.best_value)
  
  y_pred_xgb_best, y_proba_xgb_best, xgb_best_model= xgb_model_train(X_resample=X_resample, y_resample=y_resample, X_test=X_test, **study.best_params)
  print(evaluate_model(y_test, y_pred_xgb_best, y_proba_xgb_best))
  
  print("\n===== ROC =====")
  plot_roc(y_test, y_proba_xgb, y_proba_xgb_best)

  print("\n===== CONFUSION MATRIX =====")
  plot_confusion(y_test, y_pred_xgb_best, "xgb_best", "XGBoost Tuned")

  print("\n===== FEATURE IMPORTANCE =====")
  plot_feature_importance(xgb_best_model, X_train, "feature_import_best")
  
  print("\n===== OPTUNA TUNING LIGHTGBM =====")
  study = optuna.create_study(direction="maximize")
  study.optimize(lambda trial: objective_lgb(trial=trial, X_resample=X_train, y_resample=y_train, X_test=X_val, y_test=y_val), n_trials=50)
  print("Best params:", study.best_params)
  print("Best validation ROC-AUC:", study.best_value)
  
  y_pred_lgb_best, y_proba_lgb_best= lgb_model_train(X_resample=X_resample, y_resample=y_resample, X_test=X_test, **study.best_params)
  print(evaluate_model(y_test, y_pred_lgb_best, y_proba_lgb_best))
  
  print("\n===== ROC =====")
  plot_roc(y_test, y_proba_lgb, y_proba_lgb_best)

  print("\n===== CONFUSION MATRIX =====")
  plot_confusion(y_test, y_pred_lgb_best, "lgb_best", "LIGHTGBM Tuned")

if __name__ == '__main__':
  main()
  print("\n===== Task 6")
  model.main()