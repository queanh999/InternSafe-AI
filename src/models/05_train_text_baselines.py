from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# INTERNSAFE AI
# STEP 05 - TEXT BASELINE MODELS
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "model_results"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "baselines"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)


print("=" * 90)
print("INTERNSAFE AI - TEXT BASELINE TRAINING")
print("=" * 90)


print(
    f"\nTrain shape      : "
    f"{train_df.shape}"
)

print(
    f"Validation shape : "
    f"{validation_df.shape}"
)


# ------------------------------------------------------------
# 3. INPUT / TARGET
# ------------------------------------------------------------

X_train_text = (
    train_df["combined_text"]
    .fillna("")
    .astype(str)
)

X_val_text = (
    validation_df["combined_text"]
    .fillna("")
    .astype(str)
)


y_train = (
    train_df["fraudulent"]
    .astype(int)
)

y_val = (
    validation_df["fraudulent"]
    .astype(int)
)


print(
    f"\nTrain fraud rate: "
    f"{y_train.mean() * 100:.2f}%"
)

print(
    f"Validation fraud rate: "
    f"{y_val.mean() * 100:.2f}%"
)


# ============================================================
# 4. TF-IDF
# ============================================================

print("\n" + "=" * 90)
print("FITTING TF-IDF ON TRAIN ONLY")
print("=" * 90)


vectorizer = TfidfVectorizer(
    lowercase=True,

    # Word unigram + bigram
    ngram_range=(1, 2),

    # Ignore extremely rare terms
    min_df=2,

    # Ignore terms appearing in almost all documents
    max_df=0.98,

    # Prevent vocabulary becoming unnecessarily huge
    max_features=50000,

    # Helps linear models
    sublinear_tf=True,

    # Keep useful alphanumeric tokens
    token_pattern=r"(?u)\b\w\w+\b"
)


start = time.time()


X_train = vectorizer.fit_transform(
    X_train_text
)

X_val = vectorizer.transform(
    X_val_text
)


tfidf_time = time.time() - start


print(
    f"Vocabulary size : "
    f"{len(vectorizer.vocabulary_):,}"
)

print(
    f"Train matrix    : "
    f"{X_train.shape}"
)

print(
    f"Validation matrix: "
    f"{X_val.shape}"
)

print(
    f"TF-IDF time     : "
    f"{tfidf_time:.2f} seconds"
)


# Save vectorizer only as a baseline artifact
joblib.dump(
    vectorizer,
    MODEL_DIR / "text_tfidf_baseline.joblib"
)


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

results = []


def evaluate_model(
    model_name,
    model,
    X_train_matrix,
    X_validation_matrix,
    y_train_values,
    y_validation_values,
):

    print("\n" + "=" * 90)
    print(f"TRAINING: {model_name}")
    print("=" * 90)

    start_time = time.time()

    model.fit(
        X_train_matrix,
        y_train_values
    )

    train_time = (
        time.time()
        - start_time
    )


    predictions = model.predict(
        X_validation_matrix
    )


    # --------------------------------------------------------
    # Continuous fraud score
    # --------------------------------------------------------

    if hasattr(
        model,
        "predict_proba"
    ):

        fraud_scores = (
            model
            .predict_proba(
                X_validation_matrix
            )[:, 1]
        )

    elif hasattr(
        model,
        "decision_function"
    ):

        fraud_scores = (
            model
            .decision_function(
                X_validation_matrix
            )
        )

    else:

        fraud_scores = predictions


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_validation_values,
        predictions
    )

    precision_fraud = precision_score(
        y_validation_values,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall_fraud = recall_score(
        y_validation_values,
        predictions,
        pos_label=1,
        zero_division=0
    )

    f1_fraud = f1_score(
        y_validation_values,
        predictions,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_validation_values,
        predictions,
        average="macro",
        zero_division=0
    )


    try:

        roc_auc = roc_auc_score(
            y_validation_values,
            fraud_scores
        )

    except ValueError:

        roc_auc = np.nan


    try:

        pr_auc = average_precision_score(
            y_validation_values,
            fraud_scores
        )

    except ValueError:

        pr_auc = np.nan


    cm = confusion_matrix(
        y_validation_values,
        predictions
    )


    tn, fp, fn, tp = cm.ravel()


    result = {
        "model": model_name,

        "accuracy":
            accuracy,

        "fraud_precision":
            precision_fraud,

        "fraud_recall":
            recall_fraud,

        "fraud_f1":
            f1_fraud,

        "macro_f1":
            macro_f1,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        "train_seconds":
            train_time
    }


    results.append(result)


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        f"\nAccuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"Fraud Precision: "
        f"{precision_fraud:.4f}"
    )

    print(
        f"Fraud Recall   : "
        f"{recall_fraud:.4f}"
    )

    print(
        f"Fraud F1       : "
        f"{f1_fraud:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{macro_f1:.4f}"
    )

    print(
        f"ROC-AUC        : "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC         : "
        f"{pr_auc:.4f}"
    )

    print(
        f"\nTN={tn:,}  "
        f"FP={fp:,}  "
        f"FN={fn:,}  "
        f"TP={tp:,}"
    )


    print("\nClassification report:")

    print(
        classification_report(
            y_validation_values,
            predictions,
            target_names=[
                "Real",
                "Fraudulent"
            ],
            zero_division=0
        )
    )


    # --------------------------------------------------------
    # Confusion matrix figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xticks(
        [0, 1],
        ["Real", "Fraud"]
    )

    plt.yticks(
        [0, 1],
        ["Real", "Fraud"]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )


    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )


    plt.tight_layout()


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )


    plt.savefig(
        RESULT_DIR
        / f"cm_{safe_name}.png",
        dpi=200
    )

    plt.close()


    # Save baseline model
    joblib.dump(
        model,
        MODEL_DIR
        / f"{safe_name}.joblib"
    )


# ============================================================
# 6. MODEL 0 - DUMMY
# ============================================================

dummy = DummyClassifier(
    strategy="most_frequent"
)

evaluate_model(
    "Dummy Most Frequent",
    dummy,
    X_train,
    X_val,
    y_train,
    y_val
)


# ============================================================
# 7. MODEL 1 - LOGISTIC REGRESSION
# ============================================================

logistic = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    random_state=42
)

evaluate_model(
    "Logistic Regression",
    logistic,
    X_train,
    X_val,
    y_train,
    y_val
)


# ============================================================
# 8. MODEL 2 - LOGISTIC BALANCED
# ============================================================

logistic_balanced = LogisticRegression(
    max_iter=2000,
    solver="liblinear",
    class_weight="balanced",
    random_state=42
)

evaluate_model(
    "Logistic Balanced",
    logistic_balanced,
    X_train,
    X_val,
    y_train,
    y_val
)


# ============================================================
# 9. MODEL 3 - LINEAR SVM BALANCED
# ============================================================

linear_svm = LinearSVC(
    class_weight="balanced",
    random_state=42
)

evaluate_model(
    "Linear SVM Balanced",
    linear_svm,
    X_train,
    X_val,
    y_train,
    y_val
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        by="fraud_f1",
        ascending=False
    )
)


RESULT_PATH = (
    RESULT_DIR
    / "text_baseline_results.csv"
)


results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("TEXT BASELINE COMPARISON")
print("=" * 90)


display_columns = [
    "model",
    "accuracy",
    "fraud_precision",
    "fraud_recall",
    "fraud_f1",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "fp",
    "fn"
]


print(
    results_df[
        display_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("TEXT BASELINE TRAINING COMPLETED")
print("=" * 90)

print(
    f"\nResults saved to:\n"
    f"{RESULT_PATH}"
)