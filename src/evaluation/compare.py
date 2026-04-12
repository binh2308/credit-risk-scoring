import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import utils

load_dotenv()
BASE_DIR = utils.get_base_dir()
OUTPUT_PATH = BASE_DIR / Path(os.getenv("OUTPUT_PATH", "outputs/compares/"))
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

def evaluate_model(y_test, y_pred, y_proba):
    """
    Tính 4 metrics chính cho mô hình phân loại nhị phân.
    
    Args:
        y_test (array): True labels [0, 1]
        y_pred (array): Binary predictions [0, 1]
        y_proba (array): Probability scores [0.0-1.0]
    
    Returns:
        dict: {"ROC-AUC": float, "F1": float, "Recall": float, "Precision": float}
    
    Metrics:
        - ROC-AUC: Area under ROC curve (0-1). Higher is better.
        - F1: Harmonic mean of Precision & Recall (0-1). Balances both.
        - Recall: TP/(TP+FN). % of defaults caught. **Critical for banking**
        - Precision: TP/(TP+FP). Accuracy of positive predictions.
    """
    return {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "F1": f1_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred)
    }


def plot_confusion(y_test, y_pred, filename, title="Confusion Matrix"):
    """
    Vẽ confusion matrix 2x2 cho bài toán phân loại nhị phân.
    
    Args:
        y_test (array): True labels
        y_pred (array): Predictions
        filename (str): Output PNG filename with extension
        title (str): Plot title
    
    Output:
        Saves to: outputs/compares/{filename}
    
    Interpretation:
        - [0, 0] TN: Correctly predicted NOT default
        - [0, 1] FP: Falsely predicted default (lose customer)
        - [1, 0] FN: Missed actual default (lose money - critical!)
        - [1, 1] TP: Correctly predicted default
    """
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(title)
    plt.savefig(OUTPUT_PATH / filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc(y_test, proba_dict, filename="roc_curve"):
    """
    Vẽ ROC curves cho nhiều mô hình trên cùng một plot.
    
    Args:
        y_test (array): True labels
        proba_dict (dict): {"Model Name": y_proba_array, ...}
            Example: {
                "Logistic (Baseline)": [0.1, 0.8, 0.3, ...],
                "XGBoost (Tuned)": [0.2, 0.9, 0.2, ...],
                ...
            }
        filename (str): Output PNG name (without extension)
    
    Output:
        Saves to: outputs/compares/{filename}.png (dpi=300)
    
    Interpretation:
        - Curve closer to top-left = better model
        - Diagonal line = random classifier (AUC = 0.5)
        - Top-left corner = perfect classifier (AUC = 1.0)
    """
    plt.figure(figsize=(10, 8))
    
    for name, y_proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', linewidth=2)

    # Reference line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(OUTPUT_PATH / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_table(results_list, filename="model_comparison.csv"):
    """
    Lưu bảng so sánh metrics của tất cả mô hình ra CSV và in ra console.
    
    Args:
        results_list (list): List of dicts, each containing model evaluation metrics
            Example: [
                {"Model": "Logistic (Baseline)", "ROC-AUC": 0.7476, "F1": 0.4609, ...},
                {"Model": "LightGBM (Tuned)", "ROC-AUC": 0.7606, "F1": 0.5106, ...},
                ...
            ]
        filename (str): Output CSV filename
    
    Output:
        - File: outputs/compares/{filename}
        - Console: Pretty-printed table
    """
    import pandas as pd
    
    df = pd.DataFrame(results_list)
    # Reorder columns for readability
    cols = ['Model', 'ROC-AUC', 'F1', 'Recall', 'Precision']
    df = df[cols]
    
    # Save to CSV
    output_file = OUTPUT_PATH / filename
    df.to_csv(output_file, index=False, float_format='%.6f')
    
    # Print to console
    print(f"\n[INFO] Saved comparison table to: {output_file}")
    print(df.to_string(index=False))


def plot_feature_importance(model, X, filename="feature_importance.png"):
    """
    Vẽ Top 10 đặc trưng quan trọng nhất từ mô hình Boosting.
    
    Args:
        model: Fitted XGBClassifier or LGBMClassifier (with feature_importances_ attribute)
        X (DataFrame): Input features (used to get feature names)
        filename (str): Output PNG filename with extension
    
    Output:
        - File: outputs/compares/{filename}
        - Console: DataFrame of top 10 features printed
    
    Interpretation:
        - Feature at top = most important for predictions
        - Height = importance score (how much it contributes to decisions)
        - Example: late_payment_count typically dominates (>30%)
    """
    import pandas as pd
    
    # Extract importance scores
    importance = model.feature_importances_
    features = X.columns
    
    # Create DataFrame and sort
    df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values(by="importance", ascending=False)
    
    # Print top 10 to console
    print(f"\n[INFO] Top 10 Feature Importance ({filename}):")
    print(df.head(10).to_string(index=False))
    
    # Plot top 10
    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][:10], df["importance"][:10], color='steelblue')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.xlabel('Importance Score', fontsize=11)
    plt.title("Top 10 Most Important Features (XGBoost)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / filename, dpi=300, bbox_inches='tight')
    plt.close()