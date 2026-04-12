# Credit Risk Scoring v2 - Advanced Ensemble Pipeline

This project implements an end-to-end machine learning pipeline for credit risk scoring using the UCI "Default of Credit Card Clients" dataset. It leverages advanced ensemble models (LightGBM, XGBoost), SMOTE for imbalanced data handling, and Optuna for automated hyperparameter tuning.

## 🚀 Key Features

- **Robust Data Cleaning**: Handling categorical inconsistencies and payment status anomalies.
- **Advanced Feature Engineering**: Creation of intuitive financial ratios (Credit Utilization, Payment Ratio) and behavioral indicators.
- **Imbalance Handling**: Using SMOTE (Synthetic Minority Over-sampling Technique) to balance target classes (Default vs. Non-Default).
- **Automated Tuning**: Bayesian optimization via Optuna to find optimal hyperparameters for LightGBM and XGBoost.
- **Comprehensive Evaluation**: Comparison between Baseline (Logistic Regression) and Advanced Models using ROC-AUC, F1-Score, and Precision-Recall metrics.

## 🛠 Technology Stack

- **Data Processing**: `pandas`, `numpy`, `scikit-learn`
- **Imbalance Handling**: `imbalanced-learn` (SMOTE)
- **Modeling**: `lightgbm`, `xgboost`, `scikit-learn` (Logistic Regression)
- **Hyperparameter Tuning**: `optuna`
- **Visualization**: `matplotlib`, `seaborn`
- **Reporting**: `LaTeX` (source files in `reports/`)

## 📂 Project Structure

```text
credit-risk-scoring-v2/
├── data/
│   ├── raw/             # Original dataset (.xls)
│   └── processed/       # Cleaned and engineered datasets (.csv)
├── models/              # Saved model checkpoints (.pkl)
├── reports/             # Technical report (LaTeX source)
├── src/
│   ├── data_processing/ # Scripts for loading, cleaning, and balancing data
│   ├── models/          # Model definitions and training logic
│   ├── evaluation/      # Evaluation metrics and plotting logic
│   ├── utils/           # Shared utility functions
│   ├── eda.py           # Exploratory Data Analysis script
│   └── main.py          # Central execution entry point
├── requirements.txt     # Project dependencies
└── README.md            # Project overview
```

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd credit-risk-scoring-v2
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Execution Workflow

The project is designed to be executed in stages. You can run individual components or the full pipeline.

### Step 1: Data Preparation
```bash
python src/data_processing/load_data.py
python src/data_processing/data_cleaning.py
```
*Creates `credit_risk_raw.csv` and `credit_risk_cleaned.csv` respectively.*

### Step 2: Exploratory Data Analysis (Optional)
```bash
python src/eda.py
```
*Generates distribution plots and correlation heatmaps in `outputs/eda/`.*

### Step 3: Run Full Pipeline (Baseline + Advanced + Tuning)
```bash
cd src
python main.py
```
*This script performs data splitting, SMOTE resampling, trains LightGBM/XGBoost, executes Optuna tuning, and saves evaluation plots to `outputs/compares/`.*

## 📊 Results Summary

The pipeline evaluates models primarily based on **ROC-AUC** to measure the ability to distinguish between default and non-default clients. 

- **Baseline**: Logistic Regression on original imbalanced data.
- **Advanced**: XGBoost and LightGBM on SMOTE-balanced data with optimized parameters.

Detailed metrics and plots can be found in the `outputs/` directory after execution.
