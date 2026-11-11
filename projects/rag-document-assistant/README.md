# RAG Document Assistant

A practical Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents, retrieve relevant passages, and ask questions grounded in those passages.

## Features
- Upload one or more PDF files.
- Extract and split document text into overlapping chunks.
- Retrieve relevant chunks with TF-IDF and cosine similarity.
- Optionally generate answers using a locally running Ollama model.
- Show retrieved source passages so answers can be checked.
- Falls back to showing relevant context when Ollama is unavailable.

## Architecture
PDFs → text extraction → chunking → TF-IDF retrieval → relevant context → optional local LLM (Ollama).

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To enable answer generation, install [Ollama](https://ollama.com), start it, and pull a model such as `llama3.2`. Set the model name in the sidebar.

## Notes
- Retrieval is local and does not send documents to a hosted API.
- Ollama runs on your machine; model downloads require disk space.
- This is a portfolio/demo application. Verify answers against the cited source passages.
