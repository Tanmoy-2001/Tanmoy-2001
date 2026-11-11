# AI Resume & Job Match Analyzer

A resume-screening support tool that compares a candidate resume with a job description using TF-IDF similarity and transparent skill matching.

## Features
- Upload a PDF resume or paste resume text.
- Paste a target job description.
- Calculate a text similarity score using TF-IDF and cosine similarity.
- Identify matched and missing skills from an editable starter skill taxonomy.
- Show relevant terms to help tailor a resume.

## Run locally
```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```

## Method
The similarity score is a lexical baseline, not a hiring probability or an objective measure of candidate quality. Skill matching uses a transparent keyword list that can be extended for a specific role.

## Responsible use
Use this as a personal career-planning aid, not an automated hiring decision system. Review context, experience, accessibility, and transferable skills manually.
