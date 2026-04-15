import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Global_Cybersecurity_Threats_2015-2024.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ── Target columns ─────────────────────────────────────────────────────────
CLF_TARGET = "Attack Type"
REG_TARGET = "Financial Loss (in Million $)"

# ── Feature definitions (Classification) ───────────────────────────────────
NUMERIC_FEATURES_CLF = [
    "Year",
    "Financial Loss (in Million $)",
    "Number of Affected Users",
    "Incident Resolution Time (in Hours)",
    "Loss_per_User",
    "Users_per_Hour",
    "Loss_per_Hour",
    "Log_Financial_Loss",
]

CATEGORICAL_FEATURES_CLF = [
    "Country",
    "Target Industry",
    "Attack Source",
    "Security Vulnerability Type",
    "Defense Mechanism Used",
    "Year_Period",
    "Attack_Severity",
]

# ── Feature definitions (Regression) ───────────────────────────────────────
NUMERIC_FEATURES_REG = [
    "Year",
    "Number of Affected Users",
    "Incident Resolution Time (in Hours)",
    "Users_per_Hour",
]

CATEGORICAL_FEATURES_REG = [
    "Country",
    "Attack Type",
    "Target Industry",
    "Attack Source",
    "Security Vulnerability Type",
    "Defense Mechanism Used",
    "Year_Period",
]

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_DIR = "file:///" + os.path.join(BASE_DIR, "mlruns").replace("\\", "/")
