import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING & CLEANING
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    """Load the raw CSV dataset."""
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df):
    """Handle missing values and duplicates."""
    df = df.copy()
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def feature_engineering(df):
    """Create derived features: Loss_per_User, Year_Period, Attack_Severity, and more."""
    df = df.copy()

    # Loss per user (dollars)
    if "Financial Loss (in Million $)" in df.columns and "Number of Affected Users" in df.columns:
        df["Loss_per_User"] = (
            df["Financial Loss (in Million $)"] * 1e6
        ) / df["Number of Affected Users"].replace(0, np.nan)
        df["Loss_per_User"] = df["Loss_per_User"].fillna(0)
        df["Loss_per_User"] = df["Loss_per_User"].clip(upper=df["Loss_per_User"].quantile(0.99))

    # Users per hour of resolution
    if "Number of Affected Users" in df.columns and "Incident Resolution Time (in Hours)" in df.columns:
        df["Users_per_Hour"] = (
            df["Number of Affected Users"]
            / df["Incident Resolution Time (in Hours)"].replace(0, np.nan)
        )
        df["Users_per_Hour"] = df["Users_per_Hour"].fillna(0)
        df["Users_per_Hour"] = df["Users_per_Hour"].clip(upper=df["Users_per_Hour"].quantile(0.99))

    # Resolution efficiency (loss per hour of resolution)
    if "Financial Loss (in Million $)" in df.columns and "Incident Resolution Time (in Hours)" in df.columns:
        df["Loss_per_Hour"] = (
            df["Financial Loss (in Million $)"]
            / df["Incident Resolution Time (in Hours)"].replace(0, np.nan)
        )
        df["Loss_per_Hour"] = df["Loss_per_Hour"].fillna(0)
        df["Loss_per_Hour"] = df["Loss_per_Hour"].clip(upper=df["Loss_per_Hour"].quantile(0.99))

    # Log-transformed financial loss (reduces skewness)
    if "Financial Loss (in Million $)" in df.columns:
        df["Log_Financial_Loss"] = np.log1p(df["Financial Loss (in Million $)"])

    # Year period
    if "Year" in df.columns:
        bins = [2014, 2017, 2020, 2030]
        labels = ["2015-2017", "2018-2020", "2021-2030"]
        df["Year_Period"] = pd.cut(df["Year"], bins=bins, labels=labels).astype(str)

    # Attack severity buckets
    if "Financial Loss (in Million $)" in df.columns:
        loss_bins = [-0.01, 25, 50, 75, 100]
        loss_labels = ["Low", "Medium", "High", "Critical"]
        df["Attack_Severity"] = pd.cut(
            df["Financial Loss (in Million $)"], bins=loss_bins, labels=loss_labels
        ).astype(str)

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  SKLEARN PREPROCESSING PIPELINES
# ═══════════════════════════════════════════════════════════════════════════

def build_preprocessor(numeric_features, categorical_features):
    """Build a ColumnTransformer with StandardScaler + OneHotEncoder."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# ═══════════════════════════════════════════════════════════════════════════
#  PREPARE CLASSIFICATION DATA
# ═══════════════════════════════════════════════════════════════════════════

def get_classification_data():
    """Return processed train/test splits for Attack Type classification."""
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    X = df[NUMERIC_FEATURES_CLF + CATEGORICAL_FEATURES_CLF]
    y = df[CLF_TARGET]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    preprocessor = build_preprocessor(NUMERIC_FEATURES_CLF, CATEGORICAL_FEATURES_CLF)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "clf_preprocessor.pkl"))
    joblib.dump(le, os.path.join(MODELS_DIR, "clf_label_encoder.pkl"))

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "label_encoder": le,
        "feature_names": list(feature_names),
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  PREPARE REGRESSION DATA
# ═══════════════════════════════════════════════════════════════════════════

def get_regression_data():
    """Return processed train/test splits for Financial Loss regression."""
    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    X = df[NUMERIC_FEATURES_REG + CATEGORICAL_FEATURES_REG]
    y = df[REG_TARGET].values

    preprocessor = build_preprocessor(NUMERIC_FEATURES_REG, CATEGORICAL_FEATURES_REG)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "reg_preprocessor.pkl"))

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": list(feature_names),
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD SPLITS (reproducibility across pipeline steps)
# ═══════════════════════════════════════════════════════════════════════════

def save_splits(data, prefix):
    """Save processed splits to disk for reproducibility."""
    path = os.path.join(MODELS_DIR, f"{prefix}_splits.pkl")
    joblib.dump(data, path)


def load_splits(prefix):
    """Load saved splits; returns None if not found."""
    path = os.path.join(MODELS_DIR, f"{prefix}_splits.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    df = load_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    df = clean_data(df)
    df = feature_engineering(df)
    print(f"\nAfter feature engineering: {df.shape}")
    print(f"New columns: Loss_per_User, Year_Period, Attack_Severity")

    print("\n--- Classification Data ---")
    clf_data = get_classification_data()
    print(f"X_train: {clf_data['X_train'].shape}  X_test: {clf_data['X_test'].shape}")
    print(f"Classes: {list(clf_data['label_encoder'].classes_)}")

    print("\n--- Regression Data ---")
    reg_data = get_regression_data()
    print(f"X_train: {reg_data['X_train'].shape}  X_test: {reg_data['X_test'].shape}")
    print(f"Target range: [{reg_data['y_train'].min():.2f}, {reg_data['y_train'].max():.2f}]")

    print("\nPreprocessing complete! Artifacts saved to models/")
