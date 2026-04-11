from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def lgb_model_train(X_resample, y_resample, X_test, **params):
  lgb_model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
  ) if not params else LGBMClassifier(**params,random_state=42)
  
  lgb_model.fit(X_resample, y_resample)
  y_pred = lgb_model.predict(X_test)
  y_proba = lgb_model.predict_proba(X_test)[:,1]
  return y_pred, y_proba

def xgb_model_train(X_resample, y_resample, X_test, **params):
  xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    eval_metric='logloss'
  ) if not params else XGBClassifier(**params, random_state=42, eval_metric='logloss')
  
  xgb_model.fit(X_resample, y_resample)
  y_pred = xgb_model.predict(X_test)
  y_proba = xgb_model.predict_proba(X_test)[:,1]
  return y_pred, y_proba, xgb_model

def train_without_smote(X_train, y_train, X_test):
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    return y_pred, y_proba

def objective_xgb(trial, X_resample, y_resample, X_test, y_test):
  params = {
    "max_depth": trial.suggest_int("max_depth", 3, 10),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
  }
  
  _, y_proba, __= xgb_model_train(X_resample=X_resample, y_resample=y_resample,X_test=X_test, **params)
  
  score = roc_auc_score(y_test, y_proba)
  return score

def objective_lgb(trial, X_resample, y_resample, X_test, y_test):
  params = {
    "max_depth": trial.suggest_int("max_depth", 3, 10),
    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    "n_estimators": trial.suggest_int("n_estimators", 100, 400),
    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
  }
  
  _, y_proba = lgb_model_train(X_resample=X_resample, y_resample=y_resample,X_test=X_test, **params)
  
  score = roc_auc_score(y_test, y_proba)
  return score

