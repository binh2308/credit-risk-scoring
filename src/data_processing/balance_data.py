
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pandas import DataFrame


def create_sample(df: DataFrame, random_state=42):
  X = df.drop(columns=['default_payment_next_month'])
  y = df['default_payment_next_month']
  return train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train, random_state=42):
  smote = SMOTE(random_state=random_state)
  X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
  return X_resampled, y_resampled

