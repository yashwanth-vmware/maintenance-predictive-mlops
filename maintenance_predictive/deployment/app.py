import os
import io
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download, HfApi

# --- compatibility for different hub versions ---
try:
    from huggingface_hub.utils import HfHubHTTPError
except Exception:
    class HfHubHTTPError(Exception):
        pass

# ---------------------------
# Page & Environment
# ---------------------------
st.set_page_config(
    page_title="Predictive Maintenance — Engine Health",
    page_icon="⚙️",
    layout="centered"
)

# Configure via Space → Settings → Variables & secrets
MODEL_REPO = os.getenv(
    "MODEL_REPO",
    "Yashwanthsairam/engine-predictive-maintenance-xgboost"
)
MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME",
    "best_engine_model_xgb.joblib"
)
REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "model")

# ---------------------------
# Model loader (cached)
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_model(repo_id: str, filename: str, repo_type: str = "model"):
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type
        )
        return joblib.load(path)
    except Exception as e:
        files = None
        try:
            files = [f.rfilename for f in HfApi().list_repo_files(repo_id, repo_type=repo_type)]
        except Exception:
            pass

        st.error(
            "❌ Failed to load model from Hugging Face Hub.\n\n"
            f"- repo_id: `{repo_id}`\n"
            f"- filename: `{filename}`\n"
            f"- available files: {files}\n\n"
            f"Error: {e}"
        )
        raise

# ---------------------------
# UI
# ---------------------------
st.title("⚙️ Predictive Maintenance — Engine Health")
st.write("Predict whether an engine is **Healthy** or **At Risk** based on sensor readings.")

# ---------------------------
# Input form (example schema)
# ---------------------------
st.subheader("Engine Sensor Inputs")

c1, c2 = st.columns(2)

with c1:
    rpm = st.number_input("RPM", 0, 10000, 2500)
    coolant_temp = st.number_input("Coolant Temperature (°C)", -50, 200, 90)
    oil_pressure = st.number_input("Oil Pressure (psi)", 0.0, 200.0, 55.0)
    vibration = st.number_input("Vibration Level", 0.0, 100.0, 12.5)

with c2:
    fuel_rate = st.number_input("Fuel Consumption Rate", 0.0, 100.0, 15.0)
    engine_load = st.slider("Engine Load (%)", 0, 100, 65)
    ambient_temp = st.number_input("Ambient Temperature (°C)", -50, 60, 30)
    runtime_hours = st.number_input("Engine Runtime Hours", 0, 100000, 1500)

# ---------------------------
# Build input DataFrame
# ---------------------------
input_df = pd.DataFrame([{
    "RPM": rpm,
    "Coolant_Temperature": coolant_temp,
    "Oil_Pressure": oil_pressure,
    "Vibration": vibration,
    "Fuel_Rate": fuel_rate,
    "Engine_Load": engine_load,
    "Ambient_Temperature": ambient_temp,
    "Runtime_Hours": runtime_hours
}])

st.markdown("#### Input Preview")
st.dataframe(input_df, use_container_width=True)

# ---------------------------
# Load model
# ---------------------------
with st.spinner("Loading model from Hugging Face Hub…"):
    model = load_model(MODEL_REPO, MODEL_FILENAME, REPO_TYPE)
    st.success(f"Model loaded: **{MODEL_REPO} / {MODEL_FILENAME}**")

# ---------------------------
# Prediction helper
# ---------------------------
def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    preds = model.predict(df)
    proba = model.predict_proba(df)[:, 1]

    out = df.copy()
    out["failure_probability"] = proba
    out["failure_prediction"] = preds
    return out

# ---------------------------
# Actions
# ---------------------------
a, b = st.columns(2)

with a:
    if st.button("🔮 Predict Engine Health"):
        try:
            result = predict_df(input_df)
            pred = int(result.loc[0, "failure_prediction"])
            prob = result.loc[0, "failure_probability"]

            status = "⚠️ At Risk" if pred == 1 else "✅ Healthy"
            st.subheader("Prediction Result")
            st.success(f"{status} — Failure Probability: **{prob:.3f}**")

        except HfHubHTTPError as e:
            st.error(
                "Hugging Face Hub access error. "
                "If the model repo is private, add HF_TOKEN in Space secrets.\n\n"
                f"{e}"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with b:
    uploaded = st.file_uploader(
        "📦 Batch Prediction — Upload CSV (same schema, no target)",
        type=["csv"]
    )
    if uploaded and st.button("Run Batch Prediction"):
        try:
            batch_df = pd.read_csv(io.BytesIO(uploaded.read()))
            res = predict_df(batch_df)

            st.success("Batch prediction completed.")
            st.dataframe(res.head(50), use_container_width=True)

            st.download_button(
                "⬇️ Download Predictions",
                data=res.to_csv(index=False),
                file_name="engine_predictions.csv"
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

st.caption(
    "If loading fails with 404, verify **MODEL_REPO** and **MODEL_FILENAME** "
    "in Hugging Face Space settings."
)

