# Data Preparation script for Predictive Maintenance (Engine Health)

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# ----------------------------
# Hugging Face + Paths
# ----------------------------
HF_DATASET_REPO_ID = "Yashwanthsairam/maintenance-predictive-mlops-data"
HF_TOKEN = os.getenv("HF_TOKEN", "")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN is not set. In Colab, add HF_TOKEN in Secrets and export it to os.environ."
    )

api = HfApi(token=HF_TOKEN)

PROJECT_DIR = os.path.join(os.getcwd(), "maintenance_predictive")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Load dataset from HF Dataset Hub
# ----------------------------
csv_url = f"https://huggingface.co/datasets/{HF_DATASET_REPO_ID}/resolve/main/engine_data.csv"
df = pd.read_csv(csv_url)
print("✅ Dataset loaded from HF Dataset Hub:", df.shape)

# ----------------------------
# Standardize column names
# ----------------------------
df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]

# Target column: support both "Engine Condition" and "Engine_Condition"
if "Engine_Condition" in df.columns:
    target_col = "Engine_Condition"
elif "Engine_Condition" not in df.columns and "Engine_Condition" in [c.replace(" ", "_") for c in df.columns]:
    target_col = "Engine_Condition"
elif "Engine_Condition" not in df.columns and "Engine_Condition" in df.columns:
    target_col = "Engine_Condition"
elif "Engine_Condition" not in df.columns and "Engine_Condition" not in df.columns and "Engine_Condition" not in df.columns:
    # common original name -> after standardization it becomes Engine_Condition
    target_col = "Engine_Condition" if "Engine_Condition" in df.columns else None
else:
    target_col = None

# Most common after standardization for your dataset:
if target_col is None and "Engine_Condition" in df.columns:
    target_col = "Engine_Condition"

if target_col is None:
    raise KeyError(f"❌ Target column not found. Available columns: {list(df.columns)}")

# ----------------------------
# Split + Save (train/test as full rows including target)
# ----------------------------
X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_path = os.path.join(DATA_DIR, "train_engine_data.csv")
test_path  = os.path.join(DATA_DIR, "test_engine_data.csv")

pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

print("✅ Saved:", train_path, "shape:", pd.read_csv(train_path).shape)
print("✅ Saved:", test_path,  "shape:", pd.read_csv(test_path).shape)

# ----------------------------
# Upload splits back to HF Dataset Hub
# ----------------------------
for fp in [train_path, test_path]:
    api.upload_file(
        path_or_fileobj=fp,
        path_in_repo=os.path.basename(fp),
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
    )
    print("⬆️ Uploaded:", os.path.basename(fp), "to", HF_DATASET_REPO_ID)
