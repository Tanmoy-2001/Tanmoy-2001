import io
import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume & Job Match Analyzer", page_icon="🧾", layout="wide")
st.title("Resume & Job Match Analyzer")
st.caption("A transparent NLP-based comparison tool. It does not make hiring decisions.")

SKILLS = [
    "python", "java", "javascript", "php", "sql", "mysql", "postgresql",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "machine learning",
    "deep learning", "natural language processing", "nlp", "power bi", "excel",
    "data analysis", "data visualization", "html", "css", "react", "streamlit",
    "git", "github", "docker", "linux", "aws", "azure", "communication",
    "problem solving", "project management", "networking", "cybersecurity"
]

def pdf_text(upload):
    reader = PdfReader(io.BytesIO(upload.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def detect_skills(text):
    clean = normalize(text)
    return sorted(skill for skill in SKILLS if re.search(r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])", clean))

resume_file = st.file_uploader("Upload resume PDF", type=["pdf"])
resume_text_input = st.text_area("Or paste resume text", height=180)
job_text = st.text_area("Paste the job description", height=240)

if st.button("Analyze match", type="primary"):
    try:
        resume = pdf_text(resume_file) if resume_file else resume_text_input
        if not resume.strip() or not job_text.strip():
            st.error("Provide both a resume and a job description.")
        else:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            vectors = vectorizer.fit_transform([resume, job_text])
            similarity = float(cosine_similarity(vectors[0], vectors[1])[0][0])
            resume_skills = set(detect_skills(resume))
            job_skills = set(detect_skills(job_text))
            matched = sorted(resume_skills & job_skills)
            missing = sorted(job_skills - resume_skills)
            coverage = len(matched) / len(job_skills) if job_skills else 0

            a, b, c = st.columns(3)
            a.metric("Text similarity", f"{similarity:.0%}")
            b.metric("Job skill coverage", f"{coverage:.0%}" if job_skills else "N/A")
            c.metric("Matched skills", len(matched))
            st.subheader("Skills found in both")
            st.write(", ".join(matched) if matched else "No listed skills matched.")
            st.subheader("Skills in job description not detected in resume")
            st.write(", ".join(missing) if missing else "No missing skills detected from the current taxonomy.")
            st.caption("Scores are approximate text-based indicators. Keyword absence does not prove absence of a skill. Edit the SKILLS list in app.py for your target roles.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
