from pathlib import Path
import pandas as pd

def load_file(file: Path) -> pd.DataFrame:
  """Read file from Path

  Args:
      file (Path): _description_

  Returns:
      pd.DataFrame: _description_
  """
  ext = file.suffix.lstrip(".")
  return pd.read_csv(file) if ext == 'csv' else pd.read_excel(file, header=1)

def get_base_dir():
    curr = Path(__file__).resolve()
    for parent in curr.parents:
        if (parent / "data").exists():
            return parent
