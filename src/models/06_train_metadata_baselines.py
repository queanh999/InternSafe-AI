from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

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
# STEP 06 - METADATA-ONLY BASELINES
# ============================================================


# ------------------------------------------------------------
# PATHS
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
    / "metadata_baselines"
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
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)

print("=" * 90)
print("INTERNSAFE AI - METADATA-ONLY BASELINES")
print("=" * 90)

print(f"\nTrain shape      : {train_df.shape}")
print(f"Validation shape : {val_df.shape}")


# ============================================================
# 2. TARGET
# ============================================================

y_train = train_df["fraudulent"].astype(int)
y_val = val_df["fraudulent"].astype(int)


# ============================================================
# 3. NUMERIC / BINARY FEATURES
# ============================================================

numeric_features = [

    # Original binary features
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

    # Text/contact signals
    "has_url",
    "has_email",
    "has_phone_like",
    "has_html",
    "has_currency_symbol",

    "exclamation_count",
    "all_caps_word_count",

    # Text-length features
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


# ============================================================
# 4. CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "country",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]


# Department is intentionally excluded from baseline because
# EDA showed very high cardinality (~1337 categories).


feature_columns = (
    numeric_features
    + categorical_features
)


X_train = train_df[feature_columns]
X_val = val_df[feature_columns]


print(
    f"\nNumeric features     : "
    f"{len(numeric_features)}"
)

print(
    f"Categorical features : "
    f"{len(categorical_features)}"
)

print(
    f"Total raw features   : "
    f"{len(feature_columns)}"
)


# ============================================================
# 5. PREPROCESSING PIPELINE
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

            # Merge very rare categories
            min_frequency=10
        )
    ),
])


preprocessor = ColumnTransformer([
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
])


# ============================================================
# 6. EVALUATION FUNCTION
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
        ),
    ])


    start = time.time()

    pipeline.fit(
        X_train,
        y_train
    )

    train_time = (
        time.time()
        - start
    )


    predictions = pipeline.predict(
        X_val
    )


    # --------------------------------------------------------
    # Fraud score
    # --------------------------------------------------------

    if hasattr(
        pipeline,
        "predict_proba"
    ):

        scores = (
            pipeline
            .predict_proba(
                X_val
            )[:, 1]
        )

    elif hasattr(
        pipeline,
        "decision_function"
    ):

        scores = (
            pipeline
            .decision_function(
                X_val
            )
        )

    else:

        scores = predictions


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
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


    try:

        roc_auc = roc_auc_score(
            y_val,
            scores
        )

    except ValueError:

        roc_auc = np.nan


    try:

        pr_auc = average_precision_score(
            y_val,
            scores
        )

    except ValueError:

        pr_auc = np.nan


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
            precision,

        "fraud_recall":
            recall,

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
            train_time,
    })


    print(
        f"\nAccuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"Fraud Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Fraud Recall   : "
        f"{recall:.4f}"
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


    # --------------------------------------------------------
    # Confusion Matrix
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

    plt.xlabel("Predicted")
    plt.ylabel("Actual")


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
        / f"cm_metadata_{safe_name}.png",
        dpi=200
    )

    plt.close()


    # --------------------------------------------------------
    # Save complete pipeline
    # --------------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_DIR
        / f"{safe_name}.joblib"
    )


# ============================================================
# 7. MODEL 0 - DUMMY
# ============================================================

evaluate_model(
    "Dummy Metadata",
    DummyClassifier(
        strategy="most_frequent"
    )
)


# ============================================================
# 8. LOGISTIC REGRESSION
# ============================================================

evaluate_model(
    "Metadata Logistic",
    LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        random_state=42
    )
)


# ============================================================
# 9. LOGISTIC BALANCED
# ============================================================

evaluate_model(
    "Metadata Logistic Balanced",
    LogisticRegression(
        max_iter=3000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42
    )
)


# ============================================================
# 10. LINEAR SVM BALANCED
# ============================================================

evaluate_model(
    "Metadata Linear SVM Balanced",
    LinearSVC(
        class_weight="balanced",
        random_state=42
    )
)


# ============================================================
# 11. RANDOM FOREST BALANCED
# ============================================================

evaluate_model(
    "Metadata Random Forest Balanced",
    RandomForestClassifier(
        n_estimators=400,

        class_weight=
            "balanced_subsample",

        max_depth=20,

        min_samples_leaf=2,

        random_state=42,

        n_jobs=-1
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
    / "metadata_baseline_results.csv"
)


results_df.to_csv(
    RESULT_PATH,
    index=False
)


# ============================================================
# 13. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("METADATA BASELINE COMPARISON")
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
    "fn",
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
print("METADATA BASELINE TRAINING COMPLETED")
print("=" * 90)

print(
    f"\nResults saved to:\n"
    f"{RESULT_PATH}"
)