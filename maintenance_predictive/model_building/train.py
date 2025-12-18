import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

import mlflow
import mlflow.sklearn

try:
    from xgboost import XGBClassifier
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    from xgboost import XGBClassifier


def main():
    HF_TOKEN = os.getenv("HF_TOKEN", "")
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. In Colab, add HF_TOKEN in Secrets and export it to os.environ['HF_TOKEN']."
        )

    api = HfApi(token=HF_TOKEN)

    HF_DATASET_REPO_ID = "Yashwanthsairam/maintenance-predictive-mlops-data"
    TARGET_COL = "Engine_Condition"

    HF_MODEL_REPO_ID = "Yashwanthsairam/maintenance-predictive-mlops"
    REPO_TYPE = "model"

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("engine_predictive_maintenance_xgb")

    base = f"https://huggingface.co/datasets/{HF_DATASET_REPO_ID}/resolve/main"
    train_url = f"{base}/train_engine_data.csv"
    test_url  = f"{base}/test_engine_data.csv"

    train_df = pd.read_csv(train_url)
    test_df  = pd.read_csv(test_url)

    train_df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in train_df.columns]
    test_df.columns  = [c.strip().replace(" ", "_").replace("-", "_") for c in test_df.columns]

    if TARGET_COL not in train_df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found. Train columns: {list(train_df.columns)}")

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
    }

    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBClassifier")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        mlflow.log_params(best_params)

        y_pred = best_model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("test_f1", float(f1))

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        mlflow.log_dict(report, "classification_report.json")

        print("✅ Best params:", best_params)
        print("✅ Test F1:", f1)

        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/best_engine_model_xgb.joblib"
        joblib.dump(best_model, model_path)

        metrics_path = "artifacts/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({"test_f1": float(f1), "best_params": best_params}, f, indent=2)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        try:
            api.repo_info(repo_id=HF_MODEL_REPO_ID, repo_type=REPO_TYPE)
            print(f"✅ Model repo exists: {HF_MODEL_REPO_ID}")
        except RepositoryNotFoundError:
            create_repo(repo_id=HF_MODEL_REPO_ID, repo_type=REPO_TYPE, private=False, token=HF_TOKEN)
            print(f"✅ Created model repo: {HF_MODEL_REPO_ID}")

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=HF_MODEL_REPO_ID,
            repo_type=REPO_TYPE,
        )
        api.upload_file(
            path_or_fileobj=metrics_path,
            path_in_repo=os.path.basename(metrics_path),
            repo_id=HF_MODEL_REPO_ID,
            repo_type=REPO_TYPE,
        )

        print(f"✅ Uploaded model + metrics to: {HF_MODEL_REPO_ID}")


if __name__ == "__main__":
    main()
