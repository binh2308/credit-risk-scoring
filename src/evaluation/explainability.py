from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import shap
from dotenv import load_dotenv
from pandas import DataFrame

from utils import utils

load_dotenv()
BASE_DIR = utils.get_base_dir()
OUTPUT_PATH = BASE_DIR / Path(os.getenv("OUTPUT_PATH", "outputs/compares/"))
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def generate_shap_summary(
    model,
    X_train: DataFrame,
    max_display: int = 15,
    filename: str = "shap_summary",
) -> Path:
    """Generate SHAP summary plot for tree-based models (XGBoost/LightGBM).

    Args:
        model: Trained tree model exposing feature importance structure.
        X_train: Feature matrix used to explain model behavior.
        max_display: Number of top features shown in plot.
        filename: Output image name (without extension).

    Returns:
        Path to saved SHAP summary image.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_train,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()

    output_file = OUTPUT_PATH / f"{filename}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file