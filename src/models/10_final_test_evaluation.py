from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
)


# ============================================================
# INTERNSAFE AI
# STEP 10 - FINAL TEST EVALUATION
#
# IMPORTANT:
# The test set is used only here, after model selection,
# hyperparameter tuning, calibration and threshold selection.
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "calibrated"
    / "calibrated_combined_svm.joblib"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "final_test"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. PRE-COMMITTED THRESHOLDS
# ============================================================

# Selected using VALIDATION only.
#
# 0.52 = maximum Fraud F1 on validation
# 0.12 = Recall >= 75% with highest precision on validation

STRICT_THRESHOLD = 0.520
SCREENING_THRESHOLD = 0.120


# ============================================================
# 3. LOAD TEST + MODEL
# ============================================================

test_df = pd.read_csv(TEST_PATH)

model = joblib.load(MODEL_PATH)


print("=" * 100)
print("INTERNSAFE AI - FINAL TEST EVALUATION")
print("=" * 100)

print(f"\nTest shape: {test_df.shape}")

print(
    f"Test Fraud rate: "
    f"{test_df['fraudulent'].mean() * 100:.2f}%"
)


# ============================================================
# 4. FEATURES
# ============================================================

numeric_features = [

    "telecommuting",
    "has_company_logo",
    "has_questions",

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

    "has_salary",

    "has_url",
    "has_email",
    "has_phone_like",
    "has_html",
    "has_currency_symbol",

    "exclamation_count",
    "all_caps_word_count",

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


feature_columns = (

    ["combined_text"]
    + numeric_features
    + categorical_features
)


X_test = test_df[
    feature_columns
]

y_test = (
    test_df["fraudulent"]
    .astype(int)
)


# ============================================================
# 5. FINAL TEST PROBABILITIES
# ============================================================

fraud_probability = (
    model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# 6. RANKING + CALIBRATION METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    fraud_probability
)

pr_auc = average_precision_score(
    y_test,
    fraud_probability
)

brier = brier_score_loss(
    y_test,
    fraud_probability
)

ll = log_loss(
    y_test,
    fraud_probability
)


print("\n" + "=" * 100)
print("FINAL PROBABILITY QUALITY")
print("=" * 100)

print(f"ROC-AUC    : {roc_auc:.4f}")
print(f"PR-AUC     : {pr_auc:.4f}")
print(f"Brier score: {brier:.4f}")
print(f"Log loss   : {ll:.4f}")


# ============================================================
# 7. THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    name,
    threshold
):

    predictions = (
        fraud_probability
        >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    fraud_f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    fraud_f2 = fbeta_score(
        y_test,
        predictions,
        beta=2,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    print("\n" + "-" * 100)

    print(
        f"{name} "
        f"(threshold={threshold:.3f})"
    )

    print("-" * 100)

    print(
        f"Accuracy        : "
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
        f"Fraud F2       : "
        f"{fraud_f2:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{macro_f1:.4f}"
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

            y_test,

            predictions,

            target_names=[
                "Real",
                "Fraudulent"
            ],

            zero_division=0
        )
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(cm)

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

    plt.title(
        f"{name} - Final Test"
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
        name.lower()
        .replace(" ", "_")
    )


    plt.savefig(

        RESULT_DIR
        / f"confusion_matrix_{safe_name}.png",

        dpi=200
    )

    plt.close()


    return {

        "operating_point":
            name,

        "threshold":
            threshold,

        "accuracy":
            accuracy,

        "fraud_precision":
            precision,

        "fraud_recall":
            recall,

        "fraud_f1":
            fraud_f1,

        "fraud_f2":
            fraud_f2,

        "macro_f1":
            macro_f1,

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "brier_score":
            brier,

        "log_loss":
            ll,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,
    }


# ============================================================
# 8. EVALUATE THE TWO FIXED OPERATING POINTS
# ============================================================

strict_results = (
    evaluate_threshold(

        "Strict High Risk",

        STRICT_THRESHOLD
    )
)


screening_results = (
    evaluate_threshold(

        "Safety Screening",

        SCREENING_THRESHOLD
    )
)


results_df = pd.DataFrame([

    strict_results,

    screening_results
])


results_df.to_csv(

    RESULT_DIR
    / "final_test_metrics.csv",

    index=False
)


# ============================================================
# 9. THREE-LEVEL RISK SYSTEM
# ============================================================

def risk_level(probability):

    if probability >= STRICT_THRESHOLD:

        return "HIGH"

    if probability >= SCREENING_THRESHOLD:

        return "REVIEW"

    return "LOW"


risk_levels = [

    risk_level(probability)

    for probability
    in fraud_probability
]


risk_df = pd.DataFrame({

    "job_id":
        test_df["job_id"],

    "actual_fraudulent":
        y_test,

    "fraud_probability":
        fraud_probability,

    "risk_level":
        risk_levels,
})


risk_df.to_csv(

    RESULT_DIR
    / "test_predictions_with_risk.csv",

    index=False
)


risk_summary = (
    risk_df
    .groupby("risk_level")
    .agg(

        jobs=(
            "job_id",
            "count"
        ),

        fraud_jobs=(
            "actual_fraudulent",
            "sum"
        ),

        observed_fraud_rate=(
            "actual_fraudulent",
            "mean"
        ),

        avg_predicted_probability=(
            "fraud_probability",
            "mean"
        )
    )
)


risk_summary[
    "observed_fraud_rate"
] *= 100

risk_summary[
    "avg_predicted_probability"
] *= 100


risk_summary = (
    risk_summary
    .round(2)
)


print("\n" + "=" * 100)
print("THREE-LEVEL RISK SYSTEM")
print("=" * 100)

print(
    risk_summary
    .to_string()
)


risk_summary.to_csv(

    RESULT_DIR
    / "risk_level_summary.csv"
)


# ============================================================
# 10. PRECISION-RECALL CURVE
# ============================================================

precision_curve, recall_curve, _ = (
    precision_recall_curve(

        y_test,

        fraud_probability
    )
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    recall_curve,
    precision_curve
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Final Test Precision-Recall Curve"
)

plt.tight_layout()


plt.savefig(

    RESULT_DIR
    / "precision_recall_curve.png",

    dpi=200
)

plt.close()


# ============================================================
# 11. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(

    y_test,

    fraud_probability
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Final Test ROC Curve"
)

plt.tight_layout()


plt.savefig(

    RESULT_DIR
    / "roc_curve.png",

    dpi=200
)

plt.close()


# ============================================================
# 12. SAVE FINAL CONFIGURATION
# ============================================================

configuration = {

    "model":
        "Calibrated Combined Linear SVM",

    "strict_threshold":
        STRICT_THRESHOLD,

    "screening_threshold":
        SCREENING_THRESHOLD,

    "risk_levels": {

        "LOW":
            "probability < 0.12",

        "REVIEW":
            "0.12 <= probability < 0.52",

        "HIGH":
            "probability >= 0.52"
    }
}


with open(

    RESULT_DIR
    / "final_model_configuration.json",

    "w",

    encoding="utf-8"

) as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 100)
print("FINAL TEST EVALUATION COMPLETED")
print("=" * 100)

print(
    f"\nResults saved to:\n"
    f"{RESULT_DIR}"
)

print(
    "\nIMPORTANT: "
    "Do not tune the model or thresholds "
    "using these test results."
)