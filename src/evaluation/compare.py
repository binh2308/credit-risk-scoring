import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import utils

load_dotenv()
BASE_DIR = utils.get_base_dir()
OUTPUT_PATH = BASE_DIR / Path(os.getenv("OUTPUT_PATH", "outputs/compares/"))
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def evaluate_model(y_test, y_pred, y_proba):
  return {
      "ROC-AUC": roc_auc_score(y_test, y_proba),
      "F1": f1_score(y_test, y_pred),
      "Recall": recall_score(y_test, y_pred),
      "Precision": precision_score(y_test, y_pred)
  }
  
def plot_confusion(y_test, y_pred, filename, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.savefig(OUTPUT_PATH / filename)
    plt.close()

def plot_roc(y_test, y_proba_lgb, y_proba_xgb):
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

    auc_lgb = auc(fpr_lgb, tpr_lgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    plt.figure(figsize=(10, 6)) 
    plt.plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={auc_lgb:.3f})')
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={auc_xgb:.3f})')

    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "roc_curve.png")
    plt.close()

def plot_feature_importance(model, X, filename="feature_importance"):
    import pandas as pd

    importance = model.feature_importances_
    features = X.columns

    df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print(df.head(10))
    plt.figure(figsize=(10, 6)) 
    plt.barh(df["feature"][:10], df["importance"][:10])
    plt.gca().invert_yaxis()
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / f"{filename}.png")
    plt.close()