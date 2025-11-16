# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path

st.set_page_config(page_title="Telecom Customer Churn — Demo App", layout="wide")

st.title("📶 Customer Churn Prediction — Telecom (Demo App)")
st.markdown("Train classification models, visualize churn by age groups, and download the dataset/model.")

DATA_PATH = Path("data/telecom_churn_sample.csv")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

@st.cache_data
def load_data(path=DATA_PATH):
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error("Dataset not found. Run make_data.py to generate data/telecom_churn_sample.csv")
    st.stop()

df = load_data()

with st.sidebar:
    st.header("Settings")
    st.write("Rows:", len(df))
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"])
    test_size = st.slider("Test set (%)", 10, 40, 25)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    retrain = st.button("Train model")

st.subheader("Dataset preview")
st.dataframe(df.head())

# Prepare age group visualizations
st.subheader("Churn distribution by age groups")
bins = [18, 30, 40, 50, 60, 80]
labels = ["18-29","30-39","40-49","50-59","60+"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True, right=False)

churn_by_age = df.groupby("age_group")["churn"].mean().reset_index()
churn_counts = df.groupby("age_group")["churn"].sum().reset_index()

fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.bar(churn_by_age["age_group"].astype(str), churn_by_age["churn"])
ax1.set_title("Churn rate by age group")
ax1.set_xlabel("Age group")
ax1.set_ylabel("Churn rate (fraction)")
ax1.grid(axis="y", linestyle=":", linewidth=0.5)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.bar(churn_counts["age_group"].astype(str), churn_counts["churn"])
ax2.set_title("Number of churned customers by age group")
ax2.set_xlabel("Age group")
ax2.set_ylabel("Count")
ax2.grid(axis="y", linestyle=":", linewidth=0.5)
st.pyplot(fig2)

# Feature set
features = ["age","call_duration","internet_usage","complaints","monthly_charges"]
X = df[features]
y = df["churn"]

def train_model(model_name="Random Forest", test_size_pct=25, rs=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_pct/100.0, random_state=rs, stratify=y)
    if model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=rs)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=rs)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=rs)
    else:
        model = SVC(probability=True, random_state=rs)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return model, metrics, (X_test, y_test, y_pred, y_proba)

# Load pretrained model if exists
pretrained = MODEL_DIR / "model_rf.pkl"
if pretrained.exists():
    try:
        loaded = joblib.load(pretrained)
        st.sidebar.success(f"Loaded pretrained model: {pretrained.name}")
    except Exception:
        loaded = None
else:
    loaded = None

if retrain or (loaded is None):
    with st.spinner("Training the model..."):
        model, metrics, test_data = train_model(model_choice, test_size, random_state)
        joblib.dump(model, MODEL_DIR / "model_trained.pkl")
        st.success("Trained and saved model to model/model_trained.pkl")
else:
    model = loaded
    # Evaluate quickly using selected settings
    _, metrics, test_data = train_model("Random Forest", test_size, random_state)

st.subheader("Model performance")
st.write("Model:", model_choice)
st.write("Accuracy:", f'{metrics["accuracy"]:.3f}')
if metrics["roc_auc"] is not None:
    st.write("ROC AUC:", f'{metrics["roc_auc"]:.3f}')
st.text("Classification report:\n" + metrics["report"])
st.write("Confusion matrix:", metrics["confusion_matrix"])

# Downloads
def file_download_link(path: str, label: str):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{Path(path).name}">{label}</a>'
    return href

st.markdown("### Downloads")
st.markdown(file_download_link("data/telecom_churn_sample.csv", "Download sample dataset (CSV)"), unsafe_allow_html=True)
trained_path = MODEL_DIR / "model_trained.pkl"
if trained_path.exists():
    st.markdown(file_download_link(str(trained_path), "Download trained model (pickle)"), unsafe_allow_html=True)
else:
    st.write("Train the model to enable model download.")

st.markdown("---")
st.markdown("### Run locally")
st.code("""
pip install -r requirements.txt
streamlit run app.py
""", language="bash")