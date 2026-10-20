# URLGuard - AI-Based Malicious URL Detection

Classifies a URL as **Safe / Suspicious / Malicious** using ML on 27 URL features and explains the risk
indicators (IP host, brand impersonation, typosquatting, risky TLD, phishing keywords...).

## Quick start
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m backend.train         # trains + saves models/model.joblib and metrics.json
python -m backend.app           # open http://127.0.0.1:5000
```

## Use the REAL dataset (do this before the demo!)
The repo trains on **synthetic demo data** by default (so it runs instantly), which gives unrealistic ~100% scores.
1. Download Kaggle "Malicious URLs dataset" (`malicious_phish.csv`, columns `url,type`).
2. Put it at `data/malicious_phish.csv`, then run `python -m backend.train` again.
Expect roughly 90-96% accuracy; training takes a few minutes (~650k URLs).
Optionally add fresh phishing URLs from PhishTank/OpenPhish and benign ones from the Tranco list.

## How it works
`URL -> normalize -> 27 features -> model probability -> thresholds (<35% Safe, 35-70% Suspicious, >70% Malicious) -> rule-based explanations -> UI`

- Models compared: Logistic Regression, Random Forest, Gradient Boosting (best by ROC-AUC is deployed)
- Train/test split is **grouped by domain** to prevent leakage
- API: `POST /api/predict {"url": "..."}`, `GET /api/metrics`

## Tests
`pip install pytest && pytest -q tests`

## Limitations
URL-only analysis (no page content, WHOIS age or reputation feeds); new legitimate sites with odd URLs may be flagged.

## Future scope
Chrome extension, SHAP explanations, WHOIS/SSL features, batch scanning.
