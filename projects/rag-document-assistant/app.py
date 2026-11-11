import io
import requests
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("RAG Document Assistant")
st.caption("Retrieve evidence from your PDFs and optionally generate an answer with a local Ollama model.")

def extract_pages(files):
    pages = []
    for file in files:
        reader = PdfReader(io.BytesIO(file.getvalue()))
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({"source": file.name, "page": page_no, "text": text})
    return pages

def make_chunks(pages, size=1100, overlap=180):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({**page, "text": piece})
            start += max(1, size - overlap)
    return chunks

def retrieve(question, chunks, vectorizer, matrix, count):
    q = vectorizer.transform([question])
    scores = cosine_similarity(q, matrix).ravel()
    best = scores.argsort()[::-1][:count]
    return [(chunks[i], float(scores[i])) for i in best if scores[i] > 0]

def ask_ollama(question, context, model):
    prompt = (
        "Answer the question using only the context below. If the answer is not in "
        "the context, say that the documents do not provide enough information. "
        "Cite source filenames and page numbers when possible.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved passages", 1, 8, 4)
    use_ollama = st.checkbox("Generate answer with local Ollama", value=False)
    model = st.text_input("Ollama model", "llama3.2")

uploads = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if uploads:
    try:
        pages = extract_pages(uploads)
        chunks = make_chunks(pages)
        if not chunks:
            st.warning("No selectable text was found. Scanned PDFs need OCR before retrieval.")
        else:
            texts = [item["text"] for item in chunks]
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(texts)
            st.success(f"Indexed {len(pages)} pages into {len(chunks)} searchable passages.")
            question = st.text_input("Ask a question about your documents")
            if st.button("Find answer", type="primary", disabled=not question.strip()):
                results = retrieve(question, chunks, vectorizer, matrix, top_k)
                if not results:
                    st.info("No matching passages found. Try different wording.")
                else:
                    context = "\n\n".join(
                        f"[{item['source']}, page {item['page']}] {item['text']}"
                        for item, _ in results
                    )
                    if use_ollama:
                        try:
                            answer = ask_ollama(question, context, model)
                            st.subheader("Answer")
                            st.write(answer)
                        except requests.RequestException as exc:
                            st.error(f"Could not reach Ollama. Is it running? Details: {exc}")
                    else:
                        st.info("Enable Ollama in the sidebar for a generated answer. Relevant evidence is shown below.")
                    st.subheader("Retrieved evidence")
                    for (item, score) in results:
                        with st.expander(f"{item['source']} · page {item['page']} · relevance {score:.3f}"):
                            st.write(item["text"])
    except Exception as exc:
        st.error(f"Could not process the uploaded PDF: {exc}")
else:
    st.info("Upload one or more text-based PDFs to begin.")
