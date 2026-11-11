# Financial News Sentiment Analyzer

A natural-language-processing demo that classifies financial headlines and text using the pretrained FinBERT model.

## Features
- Analyze a single headline or paragraph.
- Upload a CSV containing a `text` column for batch inference.
- Display positive, negative, and neutral probabilities.
- Summarize aggregate sentiment across uploaded rows.
- Download labeled predictions.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the model checkpoint and requires internet access. Inference may be slower on CPU.

## Model
Default checkpoint: `ProsusAI/finbert`. Consult its model card and license before commercial use.

## Limitations
Financial sentiment is not a price forecast, investment recommendation, or trading signal. Headlines can be ambiguous, and model outputs should be interpreted with domain context.
