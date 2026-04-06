import pandas as pd
from pathlib import Path


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Doc file CSV da qua buoc load_data.
    """
    df = pd.read_csv(input_path)
    return df


def clean_education(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gop gia tri 0, 5, 6 vao nhom 4 (Other).
    """
    df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
    return df


def clean_marriage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gop gia tri 0 vao nhom 3 (Others).
    """
    df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)
    return df


def clean_pay_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gop gia tri -2, -1 (tra truoc han) ve 0 cho cac cot PAY_0 den PAY_6.
    """
    pay_cols = [col for col in df.columns if col.startswith("PAY_")]
    df[pay_cols] = df[pay_cols].replace([-2, -1], 0)
    return df


def verify_cleaning(df: pd.DataFrame) -> None:
    """
    Kiem tra phan phoi sau khi clean.
    """
    print("\n===== EDUCATION =====")
    print(df["EDUCATION"].value_counts().sort_index())

    print("\n===== MARRIAGE =====")
    print(df["MARRIAGE"].value_counts().sort_index())

    pay_cols = [col for col in df.columns if col.startswith("PAY_")]
    neg_found = {col: (df[col] < 0).sum() for col in pay_cols if (df[col] < 0).sum() > 0}
    if neg_found:
        print("\n[WARNING] Van con gia tri am:", neg_found)
    else:
        print("\n===== PAY columns: OK (khong con gia tri am) =====")


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Luu DataFrame ra file CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nDa luu CSV: {output_path}")


def main():
    input_path  = Path("data/processed/credit_risk_raw.csv")
    output_path = Path("data/processed/credit_risk_cleaned.csv")

    if not input_path.exists():
        print(f"Khong tim thay file: {input_path}")
        print("Hay chay load_data.py truoc.")
        return

    df = load_data(input_path)

    df = clean_education(df)
    df = clean_marriage(df)
    df = clean_pay_status(df)

    verify_cleaning(df)

    save_csv(df, output_path)


if __name__ == "__main__":
    main()
