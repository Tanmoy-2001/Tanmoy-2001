# Customer Churn Prediction & Explainability

An end-to-end machine-learning demo for customer churn classification. It supports a built-in synthetic dataset and user-uploaded CSV files.

## Features
- Train/test split with stratification when possible.
- Preprocessing for numeric and categorical columns.
- Random Forest classification pipeline.
- Accuracy, precision, recall, F1, confusion matrix, and ROC-AUC when available.
- Feature importance visualization.
- Download predictions as CSV.

## CSV format
Upload a dataset with one target column named `Churn` (case-insensitive). All other columns are treated as input features. The app handles numeric and categorical columns.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important
The built-in data is synthetic and only demonstrates the workflow. It is not evidence of real-world model performance. For a meaningful experiment, use a documented dataset, check leakage, validate class balance, and tune the decision threshold for the business objective.
