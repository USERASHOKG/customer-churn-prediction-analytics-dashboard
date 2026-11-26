from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from src.churn_pipeline import clean_data, feature_importance, predict_customer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.csv"
META_PATH = ROOT / "models" / "model_metadata.json"

st.set_page_config(page_title="Telco Churn Intelligence", page_icon="📊", layout="wide")
st.title("Telco Churn Intelligence")
st.caption("Explore customer segments, model performance, and individual churn risk.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.warning("Model artifacts are missing. Run `python train.py` before starting Streamlit.")
    st.stop()

@st.cache_data
def load_data():
    return clean_data(pd.read_csv(DATA_PATH))

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()
metrics = pd.read_csv(METRICS_PATH)
metadata = json.loads(META_PATH.read_text())

risk = df.assign(Risk=df["Churn"].map({1: "Churned", 0: "Retained"}))
col1, col2, col3, col4 = st.columns(4)
col1.metric("Customers", f"{len(df):,}")
col2.metric("Churn rate", f"{df['Churn'].mean():.1%}")
col3.metric("Best model", metadata["model"])
col4.metric("Holdout ROC-AUC", f"{metadata['roc_auc']:.3f}")

st.sidebar.header("Dashboard filters")
contract_options = sorted(df["Contract"].dropna().unique())
selected_contracts = st.sidebar.multiselect("Contract", contract_options, default=contract_options)
tenure_range = st.sidebar.slider("Tenure (months)", int(df.tenure.min()), int(df.tenure.max()),
                                 (int(df.tenure.min()), int(df.tenure.max())))
filtered = df[df["Contract"].isin(selected_contracts) & df["tenure"].between(*tenure_range)]

left, right = st.columns(2)
with left:
    st.subheader("Churn by contract")
    contract_chart = (filtered.groupby("Contract", as_index=False)["Churn"].mean()
                      .assign(ChurnRate=lambda x: x["Churn"] * 100))
    st.plotly_chart(px.bar(contract_chart, x="Contract", y="ChurnRate", labels={"ChurnRate": "Churn rate (%)"},
                           color="ChurnRate", color_continuous_scale="Reds"), use_container_width=True)
with right:
    st.subheader("Churn by tenure")
    tenure_chart = (filtered.assign(TenureBand=pd.cut(filtered["tenure"], [-1, 12, 24, 48, 72],
                                                        labels=["0–12", "13–24", "25–48", "49–72"]))
                    .groupby("TenureBand", observed=False, as_index=False)["Churn"].mean())
    tenure_chart["ChurnRate"] = tenure_chart["Churn"] * 100
    st.plotly_chart(px.bar(tenure_chart, x="TenureBand", y="ChurnRate", labels={"ChurnRate": "Churn rate (%)"},
                           color="ChurnRate", color_continuous_scale="Oranges"), use_container_width=True)

st.subheader("Model comparison")
metric_cols = ["model", "precision", "recall", "f1", "roc_auc"]
st.dataframe(metrics[metric_cols].style.format({c: "{:.3f}" for c in metric_cols[1:]}), hide_index=True,
             use_container_width=True)

st.subheader("What drives churn risk?")
importance = feature_importance(model).head(15).sort_values("importance")
st.plotly_chart(px.bar(importance, x="importance", y="feature", orientation="h",
                       labels={"feature": "Feature", "importance": "Importance"}, color="importance",
                       color_continuous_scale="Teal"), use_container_width=True)

st.subheader("Live customer risk scoring")
with st.form("prediction_form"):
    a, b, c = st.columns(3)
    gender = a.selectbox("Gender", ["Female", "Male"])
    senior = a.selectbox("Senior citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    partner = a.selectbox("Partner", ["No", "Yes"])
    dependents = a.selectbox("Dependents", ["No", "Yes"])
    tenure = b.slider("Tenure (months)", 0, 72, 12)
    phone = b.selectbox("Phone service", ["Yes", "No"])
    multiple = b.selectbox("Multiple lines", ["No phone service", "No", "Yes"])
    internet = c.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    security = c.selectbox("Online security", ["No internet service", "No", "Yes"])
    backup = c.selectbox("Online backup", ["No internet service", "No", "Yes"])
    device = c.selectbox("Device protection", ["No internet service", "No", "Yes"])
    tech = c.selectbox("Tech support", ["No internet service", "No", "Yes"])
    streaming_tv = c.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    streaming_movies = c.selectbox("Streaming movies", ["No internet service", "No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
    total = st.number_input("Total charges", 0.0, 10000.0, float(monthly * tenure), step=10.0)
    submitted = st.form_submit_button("Score customer")

if submitted:
    customer = {"gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
                "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": device,
                "TechSupport": tech, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total}
    result = predict_customer(model, customer)
    st.metric("Predicted churn probability", f"{result['probability']:.1%}")
    if result["prediction"]:
        st.error(result["label"])
    else:
        st.success(result["label"])
