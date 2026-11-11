import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Customer Churn Lab", page_icon="📉", layout="wide")
st.title("Customer Churn Prediction Lab")
st.caption("End-to-end classification workflow with transparent evaluation and downloadable predictions.")

def demo_data(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = rng.normal(72, 24, n).clip(18, 180)
    support = rng.poisson(1.3, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.58, .25, .17])
    internet = rng.choice(["Fiber", "DSL", "No"], n, p=[.52, .34, .14])
    logit = -1.1 - .035 * tenure + .018 * monthly + .65 * (contract == "Month-to-month") + .38 * support + .4 * (internet == "Fiber")
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({"TenureMonths": tenure, "MonthlyCharges": monthly.round(2),
                         "SupportTickets": support, "Contract": contract,
                         "InternetService": internet, "Churn": churn})

upload = st.file_uploader("Upload customer CSV (target column: Churn)", type=["csv"])
data = pd.read_csv(upload) if upload else demo_data()
st.write("Dataset preview", data.head())
st.caption("Using synthetic demonstration data." if upload is None else f"Loaded {len(data):,} rows from uploaded CSV.")

target = next((c for c in data.columns if c.lower() == "churn"), None)
if not target:
    st.error("The uploaded CSV must contain a target column named Churn.")
    st.stop()
if data[target].nunique() < 2:
    st.error("The target column must contain at least two classes.")
    st.stop()

X = data.drop(columns=[target]).copy()
y = data[target].copy()
if y.dtype == "object":
    y = y.astype(str).str.strip().str.lower().map({"yes": 1, "true": 1, "1": 1, "churn": 1,
                                                  "no": 0, "false": 0, "0": 0, "stay": 0}).fillna(y)
if y.nunique() != 2:
    st.error("This demo currently supports binary churn targets. Encode the target into two classes.")
    st.stop()

numeric = X.select_dtypes(include=np.number).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]
transformers = []
if numeric:
    transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                               ("scale", StandardScaler())]), numeric))
if categorical:
    transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                    ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
preprocess = ColumnTransformer(transformers)
model = Pipeline([("preprocess", preprocess),
                  ("classifier", RandomForestClassifier(n_estimators=250, class_weight="balanced",
                                                        random_state=42, min_samples_leaf=2))])
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=stratify)
model.fit(X_train, y_train)
pred = model.predict(X_test)

st.subheader("Model evaluation")
st.text(classification_report(y_test, pred, zero_division=0))
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
st.pyplot(fig)
plt.close(fig)

if hasattr(model, "predict_proba") and y.nunique() == 2:
    try:
        positive = model.predict_proba(X_test)[:, 1]
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, positive):.3f}")
    except Exception:
        pass

st.subheader("Score customer records")
scored = X.copy()
scored["PredictedChurn"] = model.predict(X)
scored["ChurnProbability"] = model.predict_proba(X)[:, 1]
st.dataframe(scored.head(30), use_container_width=True)
st.download_button("Download scored customers", scored.to_csv(index=False).encode("utf-8"),
                   "churn_predictions.csv", "text/csv")
