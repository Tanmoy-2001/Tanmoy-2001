import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Financial News Sentiment", page_icon="📈", layout="wide")
st.title("Financial News Sentiment Analyzer")
st.caption("FinBERT-based sentiment classification for financial text. Not investment advice.")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)

def classify(texts):
    model = load_model()
    results = []
    for text in texts:
        if not str(text).strip():
            results.append({"label": "empty", "score": 0.0})
            continue
        scores = model(str(text)[:4000])[0]
        best = max(scores, key=lambda item: item["score"])
        results.append({"label": best["label"], "score": float(best["score"])})
    return results

tab_one, tab_batch = st.tabs(["Single text", "Batch CSV"])
with tab_one:
    text = st.text_area("Enter a financial headline or paragraph", height=150,
                        placeholder="Example: The company reported stronger quarterly revenue...")
    if st.button("Analyze sentiment", type="primary", key="single") and text.strip():
        try:
            result = classify([text])[0]
            st.metric("Predicted sentiment", result["label"].title(), f"Confidence: {result['score']:.1%}")
        except Exception as exc:
            st.error(f"Could not load or run FinBERT: {exc}")

with tab_batch:
    file = st.file_uploader("Upload CSV with a column named 'text'", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV must contain a column named text.")
        elif st.button("Analyze all rows", type="primary"):
            try:
                results = classify(df["text"].fillna("").astype(str).tolist())
                df["sentiment"] = [r["label"] for r in results]
                df["confidence"] = [r["score"] for r in results]
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df["sentiment"].value_counts())
                st.download_button("Download results", df.to_csv(index=False).encode("utf-8"),
                                   "financial_sentiment_results.csv", "text/csv")
            except Exception as exc:
                st.error(f"Batch analysis failed: {exc}")
