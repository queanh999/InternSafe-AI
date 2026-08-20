from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import SimpleImputer

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
# STEP 07 - TEXT + METADATA MODELS
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

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
    / "combined_baselines"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)

print("=" * 90)
print("INTERNSAFE AI - TEXT + METADATA TRAINING")
print("=" * 90)

print(f"\nTrain shape      : {train_df.shape}")
print(f"Validation shape : {val_df.shape}")


y_train = (
    train_df["fraudulent"]
    .astype(int)
)

y_val = (
    val_df["fraudulent"]
    .astype(int)
)


# ============================================================
# 3. FEATURE DEFINITIONS
# ============================================================

text_feature = "combined_text"


numeric_features = [

    # Original binary metadata
    "telecommuting",
    "has_company_logo",
    "has_questions",

    # Missingness indicators
    "missing_company_profile",
    "missing_requirements",
    "missing_benefits",
    "missing_salary_range",
    "missing_department",
    "missing_employment_type",
    "missing_required_experience",
    "missing_required_education",
    "missing_industry",
    "missing_function",

    # Salary availability
    "has_salary",

    # Text / contact signals
    "has_url",
    "has_email",
    "has_phone_like",
    "has_html",
    "has_currency_symbol",

    "exclamation_count",
    "all_caps_word_count",

    # Length features
    "title_char_length",
    "title_word_length",

    "company_profile_char_length",
    "company_profile_word_length",

    "description_char_length",
    "description_word_length",

    "requirements_char_length",
    "requirements_word_length",

    "benefits_char_length",
    "benefits_word_length",
]


categorical_features = [
    "country",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]


all_features = (
    [text_feature]
    + numeric_features
    + categorical_features
)


X_train = train_df[all_features]
X_val = val_df[all_features]


print(
    f"\nNumeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)

print(
    "Text feature         : combined_text"
)


# ============================================================
# 4. TEXT TRANSFORMER
# ============================================================

text_transformer = TfidfVectorizer(

    lowercase=True,

    ngram_range=(1, 2),

    min_df=2,

    max_df=0.98,

    max_features=50000,

    sublinear_tf=True,

    token_pattern=r"(?u)\b\w\w+\b",
)


# ============================================================
# 5. NUMERIC TRANSFORMER
# ============================================================

numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler(
            with_mean=False
        )
    ),
])


# ============================================================
# 6. CATEGORICAL TRANSFORMER
# ============================================================

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10
        )
    ),
])


# ============================================================
# 7. COMBINED PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "text",
            text_transformer,
            text_feature
        ),

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
    ],

    remainder="drop"
)


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

results = []


def evaluate_model(
    model_name,
    classifier
):

    print("\n" + "=" * 90)
    print(f"TRAINING: {model_name}")
    print("=" * 90)


    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )
    ])


    start = time.time()


    pipeline.fit(
        X_train,
        y_train
    )


    train_seconds = (
        time.time()
        - start
    )


    predictions = (
        pipeline.predict(
            X_val
        )
    )


    # ========================================================
    # SCORES
    # ========================================================

    if hasattr(
        pipeline,
        "predict_proba"
    ):

        fraud_scores = (
            pipeline
            .predict_proba(
                X_val
            )[:, 1]
        )

    else:

        fraud_scores = (
            pipeline
            .decision_function(
                X_val
            )
        )


    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    fraud_precision = precision_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    fraud_recall = recall_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    fraud_f1 = f1_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        fraud_scores
    )

    pr_auc = average_precision_score(
        y_val,
        fraud_scores
    )


    cm = confusion_matrix(
        y_val,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    results.append({

        "model":
            model_name,

        "accuracy":
            accuracy,

        "fraud_precision":
            fraud_precision,

        "fraud_recall":
            fraud_recall,

        "fraud_f1":
            fraud_f1,

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
            train_seconds,
    })


    # ========================================================
    # PRINT
    # ========================================================

    print(
        f"\nAccuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"Fraud Precision: "
        f"{fraud_precision:.4f}"
    )

    print(
        f"Fraud Recall   : "
        f"{fraud_recall:.4f}"
    )

    print(
        f"Fraud F1       : "
        f"{fraud_f1:.4f}"
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
        f"\nTN={tn:,} "
        f"FP={fp:,} "
        f"FN={fn:,} "
        f"TP={tp:,}"
    )


    print("\nClassification report:")

    print(
        classification_report(
            y_val,
            predictions,

            target_names=[
                "Real",
                "Fraudulent"
            ],

            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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
        / f"cm_combined_{safe_name}.png",
        dpi=200
    )

    plt.close()


    # ========================================================
    # SAVE FULL PIPELINE
    # ========================================================

    joblib.dump(
        pipeline,

        MODEL_DIR
        / f"{safe_name}.joblib"
    )


# ============================================================
# 9. MODEL 1 - LOGISTIC
# ============================================================

evaluate_model(

    "Combined Logistic",

    LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        random_state=42
    )
)


# ============================================================
# 10. MODEL 2 - LOGISTIC BALANCED
# ============================================================

evaluate_model(

    "Combined Logistic Balanced",

    LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    )
)


# ============================================================
# 11. MODEL 3 - LINEAR SVM BALANCED
# ============================================================

evaluate_model(

    "Combined Linear SVM Balanced",

    LinearSVC(
    class_weight="balanced",
    C=1.0,
    max_iter=20000,
    random_state=42
)
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        "fraud_f1",
        ascending=False
    )
)


RESULT_PATH = (
    RESULT_DIR
    / "combined_baseline_results.csv"
)


results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# 13. DISPLAY
# ============================================================

print("\n" + "=" * 90)
print("TEXT + METADATA COMPARISON")
print("=" * 90)


columns_to_show = [

    "model",

    "accuracy",

    "fraud_precision",

    "fraud_recall",

    "fraud_f1",

    "macro_f1",

    "roc_auc",

    "pr_auc",

    "fp",

    "fn",
]


print(
    results_df[
        columns_to_show
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("COMBINED TRAINING COMPLETED")
print("=" * 90)


print(
    f"\nResults saved to:\n"
    f"{RESULT_PATH}"
)