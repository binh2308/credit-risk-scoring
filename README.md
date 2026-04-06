# Credit Risk Scoring v2 - Advanced Ensemble Pipeline

This project builds an advanced credit risk scoring pipeline using state-of-the-art ensemble techniques (LightGBM, XGBoost) and optimized feature engineering based on the UCI Default of Credit Card Clients dataset.

## References & Methodology

### Ensemble Methodology

- **Innovations in Credit Default Prediction (2024)**: [arXiv:2402.17979](https://arxiv.org/pdf/2402.17979)
  Justifies why Boosting models like LightGBM and XGBoost remain superior for credit default tasks compared to deep learning or traditional linear models.

### Categorical Feature Handling

- **XGBoost Categorical Support**: [Documentation](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html) - Native support for categorical variables.
- **LightGBM Advanced Topics**: [Documentation](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html) - Native categorical splitting often outperforms one-hot encoding.
- **CatBoost Features**: [Documentation](https://catboost.ai/docs/en/features/categorical-features) - Optimized internal handling of categorical data.

### Benchmarks & Structures

- **Kaggle Benchmark**: [Default of Credit Card Clients - Predictive Models](https://www.kaggle.com/code/gpreda/default-of-credit-card-clients-predictive-models)
- **GitHub reference**: [arashshams/Credit_Card_Customer_Default](https://github.com/arashshams/Credit_Card_Customer_Default)

## Getting Started

1. Create virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Setup output directories:

   ```bash
   mkdir -p outputs/eda
   ```

4. Run the pipeline stages in `src/`.

## Proposed Models

- Logistic Regression (Baseline)
- Random Forest
- LightGBM (Native Categorical)
- XGBoost (Native Categorical)
