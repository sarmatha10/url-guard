"""
URLGuard - Model Training

Run from the project root:
    python -m backend.train

Uses the real malicious_phish.csv dataset when available.
Adds representative legitimate homepage examples.
Compares three models and saves the best model.
"""

import json
import os
import random
import string

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backend.features import (
    FEATURE_NAMES,
    registered_domain,
    vectorize,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_REAL = os.path.join(
    ROOT, "data", "malicious_phish.csv"
)

DATA_DEMO = os.path.join(
    ROOT, "data", "demo_urls.csv"
)

MODEL_DIR = os.path.join(ROOT, "models")

MODEL_PATH = os.path.join(
    MODEL_DIR, "model.joblib"
)

METRICS_PATH = os.path.join(
    MODEL_DIR, "metrics.json"
)

random.seed(42)
np.random.seed(42)


# --------------------------------------------------
# SYNTHETIC DEMO DATA
# Used only when the real dataset is unavailable.
# --------------------------------------------------

def _rand(n, chars=string.ascii_lowercase + string.digits):
    return "".join(random.choice(chars) for _ in range(n))


def make_demo_dataset(n=4000):

    good = [
        "google.com",
        "wikipedia.org",
        "github.com",
        "amazon.in",
        "flipkart.com",
        "nytimes.com",
        "stackoverflow.com",
        "bbc.co.uk",
        "linkedin.com",
        "microsoft.com",
        "apple.com",
        "youtube.com",
        "irctc.co.in",
        "hdfcbank.com",
        "sbi.co.in",
        "coursera.org",
        "medium.com",
    ]

    words = [
        "news", "about", "products", "blog", "docs",
        "careers", "help", "search", "store", "contact",
    ]

    brands = [
        "paypal", "amazon", "google", "netflix",
        "sbi", "hdfc", "microsoft", "apple",
    ]

    keywords = [
        "login", "verify", "secure", "update",
        "account", "confirm", "wallet", "bank",
    ]

    tlds = [
        "xyz", "top", "tk", "ml", "click",
        "icu", "buzz", "work",
    ]

    rows = []

    # Generate legitimate-looking demo URLs.
    for _ in range(n):

        if random.random() < 0.15:
            domain = random.choice(good)
        else:
            domain = (
                f"{random.choice(words)}{_rand(4)}."
                f"{random.choice(['com', 'org', 'in', 'net'])}"
            )

        path = "/".join(
            random.choice(words)
            for _ in range(random.randint(0, 3))
        )

        url = (
            f"{random.choice(['https', 'https', 'http'])}://"
            f"{'www.' if random.random() < 0.5 else ''}"
            f"{domain}/{path}"
        )

        rows.append((url, 0))

    # Generate suspicious-looking demo URLs.
    for _ in range(n):

        brand = random.choice(brands)
        keyword = random.choice(keywords)
        tld = random.choice(tlds)

        kind = random.randint(0, 5)

        if kind == 0:
            url = (
                f"http://{brand}-{keyword}.{_rand(6)}."
                f"{tld}/{keyword}/{_rand(8)}"
            )

        elif kind == 1:
            url = (
                f"http://{random.randint(11, 223)}."
                f"{random.randint(0, 255)}."
                f"{random.randint(0, 255)}."
                f"{random.randint(1, 254)}/"
                f"{keyword}.php?id={_rand(10)}"
            )

        elif kind == 2:
            url = (
                f"https://secure.{keyword}.{brand}."
                f"{_rand(7)}.{tld}/{keyword}-{_rand(5)}"
                f"?session={_rand(20)}"
            )

        elif kind == 3:
            url = (
                f"http://{brand[:-1]}"
                f"{random.choice('01l')}{brand[-1:]}.com/{keyword}"
            )

        elif kind == 4:
            url = (
                f"http://{_rand(12)}.{tld}:"
                f"{random.choice([8080, 8888, 4443])}/"
                f"{_rand(15)}.exe"
            )

        else:
            url = (
                f"http://{brand}.com@{_rand(8)}."
                f"{tld}/{keyword}/{_rand(6)}"
            )

        rows.append((url, 1))

    df = pd.DataFrame(
        rows,
        columns=["url", "label"],
    )

    df = df.sample(frac=1, random_state=42)

    os.makedirs(os.path.dirname(DATA_DEMO), exist_ok=True)

    df.to_csv(DATA_DEMO, index=False)

    return df


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_data():

    if os.path.exists(DATA_REAL):

        df = pd.read_csv(DATA_REAL)

        df["label"] = (
            df["type"].astype(str).str.lower() != "benign"
        ).astype(int)

        df = (
            df[["url", "label"]]
            .dropna()
            .drop_duplicates(subset="url")
            .reset_index(drop=True)
        )

        print(f"Using REAL dataset: {len(df)} URLs")

        return df, "kaggle_malicious_phish"

    print(
        "Real dataset not found. "
        "Using synthetic demo data."
    )

    return make_demo_dataset(), "synthetic_demo"


# --------------------------------------------------
# ADD LEGITIMATE HOMEPAGE EXAMPLES
# --------------------------------------------------

def add_homepage_examples(df):

    homepage_domains = [
        "google.com",
        "wikipedia.org",
        "github.com",
        "microsoft.com",
        "apple.com",
        "youtube.com",
        "amazon.com",
        "amazon.in",
        "linkedin.com",
        "stackoverflow.com",
        "bbc.com",
        "reddit.com",
        "python.org",
        "mozilla.org",
        "cloudflare.com",
        "huggingface.co",
        "coursera.org",
        "medium.com",
        "flipkart.com",
        "sbi.co.in",
        "hdfcbank.com",
        "irctc.co.in",
        "developer.mozilla.org",
        "docs.python.org",
    ]

    rows = []

    for domain in homepage_domains:

        for url in [
            f"https://{domain}",
            f"https://{domain}/",
            f"http://{domain}/",
        ]:

            rows.append({
                "url": url,
                "label": 0,
            })

    homepage_df = pd.DataFrame(rows)

    # Avoid changing the labels of URLs already in the dataset.
    combined = pd.concat(
        [df, homepage_df],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset="url",
        keep="first",
    ).reset_index(drop=True)

    print(
        "Homepage examples added before duplicate removal:",
        len(homepage_df),
    )

    return combined


# --------------------------------------------------
# MODEL TRAINING AND EVALUATION
# --------------------------------------------------

def main():

    # Load the original dataset.
    df, source = load_data()

    original_size = len(df)

    # Add representative homepage examples.
    df = add_homepage_examples(df)

    print("Original dataset size:", original_size)
    print("Training dataset size:", len(df))

    # Extract features.
    print("\nExtracting URL features...")

    X = np.array(
        [vectorize(url) for url in df["url"]],
        dtype=float,
    )

    y = df["label"].values

    # Split by registered domain to reduce domain leakage.
    groups = df["url"].map(registered_domain).values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups)
    )

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    print("Training samples:", len(train_indices))
    print("Testing samples:", len(test_indices))

    # Models to compare.
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        ),

        "Gradient Boosting": HistGradientBoostingClassifier(
            random_state=42,
        ),
    }

    results = {}

    best_name = None
    best_auc = -1
    best_model = None

    print("\nTraining and comparing models...\n")

    for name, model in models.items():

        print(f"Training {name}...")

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        auc = roc_auc_score(
            y_test,
            probabilities,
        )

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),

            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

            "roc_auc": float(auc),

            "confusion_matrix": confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            ).tolist(),
        }

        results[name] = metrics

        print(
            f"{name:22s} "
            f"Accuracy={metrics['accuracy']:.3f} "
            f"Precision={metrics['precision']:.3f} "
            f"Recall={metrics['recall']:.3f} "
            f"F1={metrics['f1']:.3f} "
            f"AUC={auc:.3f}"
        )

        if auc > best_auc:

            best_auc = auc
            best_name = name
            best_model = model

    # Save the selected model and metrics.
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    metrics_output = {
        "dataset": source,
        "train_size": int(len(train_indices)),
        "test_size": int(len(test_indices)),
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "results": results,
        "notes": [
            "Homepage examples were added to improve training coverage.",
            "Model predictions are not a guarantee that a URL is safe.",
            "Evaluation results do not guarantee correct predictions on unseen URLs.",
        ],
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_output,
            file,
            indent=2,
        )

    print("\n------------------------------------")
    print("Training completed!")
    print("Best model:", best_name)
    print("Best ROC-AUC:", round(best_auc, 4))
    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)
    print("------------------------------------")


if __name__ == "__main__":
    main()