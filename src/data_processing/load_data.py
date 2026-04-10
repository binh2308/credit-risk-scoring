import pandas as pd
from pathlib import Path


def load_excel_file(file_path: Path) -> pd.DataFrame:
    """
    Doc file Excel .xls.
    """
    df = pd.read_excel(file_path, header=1)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(col).strip().replace(" ", "_") for col in df.columns]
    return df


def basic_check(df: pd.DataFrame) -> None:
    print("\n===== SHAPE =====")
    print(df.shape)

    print("\n===== COLUMNS =====")
    print(df.columns.tolist())

    print("\n===== HEAD =====")
    print(df.head())

    print("\n===== INFO =====")
    df.info()

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DUPLICATED ROWS =====")
    print(df.duplicated().sum())

    target_col = "default_payment_next_month"
    if target_col in df.columns:
        print("\n===== TARGET DISTRIBUTION =====")
        print(df[target_col].value_counts())


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\nDa luu CSV: {output_path}")


def main():
    input_path = Path("data/raw/input_credit_card_clients.xls")
    output_path = Path("data/processed/credit_risk_raw.csv")

    if not input_path.exists():
        print(f"Khong tim thay file: {input_path}")
        return

    try:
        df = load_excel_file(input_path)
    except Exception as e:
        print("Khong doc duoc file Excel:", e)
        return

    df = clean_column_names(df)

    basic_check(df)

    save_csv(df, output_path)


if __name__ == "__main__":
    main()