import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

DATA_PATH = Path("data/processed/credit_risk_cleaned.csv")
OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "default_payment_next_month"


def load_data():
    df = pd.read_csv(DATA_PATH)

    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default_payment_next_month"})

    return df


def basic_overview(df):
    print("\n===== SHAPE =====")
    print(df.shape)

    print("\n===== HEAD =====")
    print(df.head())

    print("\n===== DTYPES =====")
    print(df.dtypes)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\n===== DESCRIBE =====")
    print(df.describe().T)

    summary_df = df.describe().T
    summary_df["missing_count"] = df.isnull().sum()
    summary_df["missing_ratio"] = df.isnull().mean() * 100
    summary_df.to_csv(OUTPUT_DIR / "eda_summary_statistics.csv")


def plot_target_distribution(df):
    print("\n===== TARGET VALUE COUNTS =====")
    print(df[TARGET].value_counts())

    print("\n===== TARGET RATIO (%) =====")
    target_ratio = df[TARGET].value_counts(normalize=True).sort_index() * 100
    print(target_ratio)

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=TARGET, data=df)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')
    plt.title("Distribution of Target Variable")
    plt.xlabel("Default Payment Next Month")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution_countplot.png", dpi=300)
    plt.show()

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x=target_ratio.index, y=target_ratio.values)
    plt.title("Target Class Percentage")
    plt.xlabel("Default Payment Next Month")
    plt.ylabel("Percentage (%)")
    for i, v in enumerate(target_ratio.values):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution_percentage.png", dpi=300)
    plt.show()


def analyze_categorical_features(df):
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]

    for col in cat_cols:
        if col in df.columns:
            print(f"\n===== {col} VALUE COUNTS =====")
            print(df[col].value_counts(dropna=False).sort_index())

            plt.figure(figsize=(7, 4))
            sns.countplot(x=col, data=df)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"{col.lower()}_distribution.png", dpi=300)
            plt.show()

            plt.figure(figsize=(7, 4))
            sns.barplot(x=col, y=TARGET, data=df, estimator=np.mean)
            plt.title(f"Default Rate by {col}")
            plt.ylabel("Default Rate")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"default_rate_by_{col.lower()}.png", dpi=300)
            plt.show()


def analyze_numeric_features(df):
    num_cols = [
        "LIMIT_BAL", "AGE",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    for col in num_cols:
        if col in df.columns:
            plt.figure(figsize=(7, 4))
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f"Histogram of {col}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"hist_{col.lower()}.png", dpi=300)
            plt.show()

            plt.figure(figsize=(7, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"box_{col.lower()}.png", dpi=300)
            plt.show()


def correlation_analysis(df):
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - Full Numeric Features")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap_full.png", dpi=300)
    plt.show()

    bill_amt_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    available_bill_cols = [col for col in bill_amt_cols if col in df.columns]

    if len(available_bill_cols) >= 2:
        bill_corr = df[available_bill_cols].corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(bill_corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap of BILL_AMT Features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "bill_amt_heatmap.png", dpi=300)
        plt.show()

        threshold = 0.90
        print("\n===== HIGHLY CORRELATED BILL_AMT PAIRS =====")
        for i in range(len(available_bill_cols)):
            for j in range(i + 1, len(available_bill_cols)):
                col1 = available_bill_cols[i]
                col2 = available_bill_cols[j]
                corr_value = df[col1].corr(df[col2])
                if abs(corr_value) > threshold:
                    print(f"{col1} - {col2}: {corr_value:.4f}")

    if TARGET in corr.columns:
        target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda x: abs(x), ascending=False)
        print("\n===== TOP CORRELATION WITH TARGET =====")
        print(target_corr.head(15))


def main():
    df = load_data()
    '''
    Muốn xem phần nào thì bỏ comment phần đó, vì có thể sẽ tốn thời gian chạy nếu chạy hết tất cả các hàm EDA.
    '''
    basic_overview(df)   ## Mô tả tổng quan về dữ liệu, bao gồm shape, head, dtypes, missing values, và thống kê mô tả.
    # plot_target_distribution(df)   ## Phân tích biến mục tiêu
    # analyze_categorical_features(df)      ##  Phân tích các biến quan trọng
    # analyze_numeric_features(df)     ## Các biến numeric nên xem histogram / boxplot
    # correlation_analysis(df)     ## Heatmap tương quan tổng thể và heatmap riêng cho các biến BILL_AMT


if __name__ == "__main__":
    main()