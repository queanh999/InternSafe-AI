from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)

from sklearn.model_selection import (
    StratifiedGroupKFold,
)

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
)


# ============================================================
# INTERNSAFE AI
# STEP 09 - CALIBRATION + THRESHOLD OPTIMIZATION
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

SVM_PATH = (
    PROJECT_ROOT
    / "models"
    / "tuned"
    / "best_combined_svm.joblib"
)

LOGISTIC_PATH = (
    PROJECT_ROOT
    / "models"
    / "tuned"
    / "best_combined_logistic.joblib"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "threshold_tuning"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "calibrated"
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
# 2. LOAD DATA + MODELS
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)

best_svm = joblib.load(SVM_PATH)
best_logistic = joblib.load(LOGISTIC_PATH)


print("=" * 95)
print("INTERNSAFE AI - CALIBRATION + THRESHOLD TUNING")
print("=" * 95)

print(f"\nTrain      : {train_df.shape}")
print(f"Validation : {val_df.shape}")


# ============================================================
# 3. FEATURES
# ============================================================

text_feature = "combined_text"

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


all_features = (
    [text_feature]
    + numeric_features
    + categorical_features
)


X_train = train_df[all_features]
X_val = val_df[all_features]

y_train = train_df[
    "fraudulent"
].astype(int)

y_val = val_df[
    "fraudulent"
].astype(int)

groups = train_df[
    "content_group"
]


# ============================================================
# 4. GROUP-AWARE CALIBRATION SPLITS
# ============================================================

group_cv = StratifiedGroupKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


cv_splits = list(

    group_cv.split(

        X_train,

        y_train,

        groups=groups
    )
)


print(
    f"\nCalibration CV folds: "
    f"{len(cv_splits)}"
)


# ============================================================
# 5. CALIBRATE SVM
# ============================================================

print("\n" + "=" * 95)
print("CALIBRATING TUNED SVM")
print("=" * 95)


calibrated_svm = CalibratedClassifierCV(

    estimator=best_svm,

    method="sigmoid",

    cv=cv_splits,

    # calibration is learned from CV predictions;
    # final estimator is fit using all training data
    ensemble=False,

    n_jobs=-1,
)


calibrated_svm.fit(
    X_train,
    y_train
)


joblib.dump(

    calibrated_svm,

    MODEL_DIR
    / "calibrated_combined_svm.joblib"
)


print(
    "\nSaved calibrated SVM."
)


# ============================================================
# 6. VALIDATION PROBABILITIES
# ============================================================

svm_prob = (
    calibrated_svm
    .predict_proba(
        X_val
    )[:, 1]
)


logistic_prob = (
    best_logistic
    .predict_proba(
        X_val
    )[:, 1]
)


# ============================================================
# 7. PROBABILITY QUALITY
# ============================================================

def probability_metrics(
    model_name,
    probabilities
):

    return {

        "model":
            model_name,

        "roc_auc":
            roc_auc_score(
                y_val,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_val,
                probabilities
            ),

        "brier_score":
            brier_score_loss(
                y_val,
                probabilities
            ),

        "log_loss":
            log_loss(
                y_val,
                probabilities
            ),
    }


probability_results = pd.DataFrame([

    probability_metrics(
        "Calibrated SVM",
        svm_prob
    ),

    probability_metrics(
        "Tuned Logistic",
        logistic_prob
    ),
])


print("\n" + "=" * 95)
print("PROBABILITY QUALITY")
print("=" * 95)

print(
    probability_results
    .round(4)
    .to_string(
        index=False
    )
)


probability_results.to_csv(

    RESULT_DIR
    / "probability_quality.csv",

    index=False
)


# ============================================================
# 8. THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)


    cm = confusion_matrix(
        y_val,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    fraud_f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    fraud_f2 = fbeta_score(
        y_val,
        predictions,
        beta=2,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )


    return {

        "threshold":
            threshold,

        "accuracy":
            accuracy_score(
                y_val,
                predictions
            ),

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

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,

        "predicted_fraud":
            int(
                predictions.sum()
            )
    }


# ============================================================
# 9. SWEEP THRESHOLDS
# ============================================================

thresholds = np.arange(
    0.05,
    0.951,
    0.005
)


svm_threshold_results = pd.DataFrame([

    calculate_threshold_metrics(
        svm_prob,
        threshold
    )

    for threshold in thresholds
])


logistic_threshold_results = pd.DataFrame([

    calculate_threshold_metrics(
        logistic_prob,
        threshold
    )

    for threshold in thresholds
])


svm_threshold_results.to_csv(

    RESULT_DIR
    / "svm_threshold_sweep.csv",

    index=False
)


logistic_threshold_results.to_csv(

    RESULT_DIR
    / "logistic_threshold_sweep.csv",

    index=False
)


# ============================================================
# 10. SELECT CANDIDATE THRESHOLDS
# ============================================================

def select_threshold_candidates(
    table
):

    # --------------------------------------------------------
    # A. Highest Fraud F1
    # --------------------------------------------------------

    best_f1 = (
        table
        .sort_values(
            [
                "fraud_f1",
                "fraud_precision"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )


    # --------------------------------------------------------
    # B. Highest F2
    # F2 gives more importance to recall
    # --------------------------------------------------------

    best_f2 = (
        table
        .sort_values(
            [
                "fraud_f2",
                "fraud_precision"
            ],
            ascending=[
                False,
                False
            ]
        )
        .iloc[0]
    )


    # --------------------------------------------------------
    # C. Among thresholds with Recall >= 75%,
    # choose the one with highest precision
    # --------------------------------------------------------

    recall_candidates = table[
        table[
            "fraud_recall"
        ] >= 0.75
    ]


    if len(
        recall_candidates
    ) > 0:

        recall75 = (
            recall_candidates
            .sort_values(
                [
                    "fraud_precision",
                    "fraud_f1"
                ],
                ascending=[
                    False,
                    False
                ]
            )
            .iloc[0]
        )

    else:

        recall75 = best_f2


    return {
        "best_f1":
            best_f1,

        "best_f2":
            best_f2,

        "recall_at_least_075":
            recall75,
    }


svm_candidates = (
    select_threshold_candidates(
        svm_threshold_results
    )
)


logistic_candidates = (
    select_threshold_candidates(
        logistic_threshold_results
    )
)


# ============================================================
# 11. PRINT CANDIDATES
# ============================================================

def print_candidate(
    model_name,
    candidate_name,
    row
):

    print(
        f"\n{model_name} "
        f"- {candidate_name}"
    )

    print(
        f"Threshold       : "
        f"{row['threshold']:.3f}"
    )

    print(
        f"Fraud Precision: "
        f"{row['fraud_precision']:.4f}"
    )

    print(
        f"Fraud Recall   : "
        f"{row['fraud_recall']:.4f}"
    )

    print(
        f"Fraud F1       : "
        f"{row['fraud_f1']:.4f}"
    )

    print(
        f"Fraud F2       : "
        f"{row['fraud_f2']:.4f}"
    )

    print(
        f"Macro F1       : "
        f"{row['macro_f1']:.4f}"
    )

    print(
        f"FP={int(row['fp'])}  "
        f"FN={int(row['fn'])}  "
        f"TP={int(row['tp'])}"
    )


print("\n" + "=" * 95)
print("SVM THRESHOLD CANDIDATES")
print("=" * 95)

for name, row in (
    svm_candidates.items()
):

    print_candidate(
        "Calibrated SVM",
        name,
        row
    )


print("\n" + "=" * 95)
print("LOGISTIC THRESHOLD CANDIDATES")
print("=" * 95)

for name, row in (
    logistic_candidates.items()
):

    print_candidate(
        "Tuned Logistic",
        name,
        row
    )


# ============================================================
# 12. SAVE CANDIDATE TABLE
# ============================================================

candidate_rows = []


for model_name, candidates in [

    (
        "Calibrated SVM",
        svm_candidates
    ),

    (
        "Tuned Logistic",
        logistic_candidates
    ),
]:

    for candidate_name, row in (
        candidates.items()
    ):

        candidate_rows.append({

            "model":
                model_name,

            "selection_rule":
                candidate_name,

            **row.to_dict()
        })


candidate_table = pd.DataFrame(
    candidate_rows
)


candidate_table.to_csv(

    RESULT_DIR
    / "threshold_candidates.csv",

    index=False
)


# ============================================================
# 13. THRESHOLD PERFORMANCE CURVE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(

    svm_threshold_results[
        "threshold"
    ],

    svm_threshold_results[
        "fraud_precision"
    ],

    label="Precision"
)


plt.plot(

    svm_threshold_results[
        "threshold"
    ],

    svm_threshold_results[
        "fraud_recall"
    ],

    label="Recall"
)


plt.plot(

    svm_threshold_results[
        "threshold"
    ],

    svm_threshold_results[
        "fraud_f1"
    ],

    label="F1"
)


plt.plot(

    svm_threshold_results[
        "threshold"
    ],

    svm_threshold_results[
        "fraud_f2"
    ],

    label="F2"
)


plt.xlabel(
    "Probability threshold"
)

plt.ylabel(
    "Metric value"
)

plt.title(
    "Calibrated SVM - Threshold Performance"
)

plt.legend()

plt.tight_layout()


plt.savefig(

    RESULT_DIR
    / "svm_threshold_performance.png",

    dpi=200
)

plt.close()


# ============================================================
# 14. CALIBRATION CURVE
# ============================================================

svm_true, svm_pred = (
    calibration_curve(

        y_val,

        svm_prob,

        n_bins=10,

        strategy="quantile"
    )
)


log_true, log_pred = (
    calibration_curve(

        y_val,

        logistic_prob,

        n_bins=10,

        strategy="quantile"
    )
)


plt.figure(
    figsize=(7, 6)
)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Perfect calibration"
)


plt.plot(

    svm_pred,

    svm_true,

    marker="o",

    label="Calibrated SVM"
)


plt.plot(

    log_pred,

    log_true,

    marker="o",

    label="Tuned Logistic"
)


plt.xlabel(
    "Mean predicted probability"
)

plt.ylabel(
    "Observed fraud frequency"
)

plt.title(
    "Probability Calibration on Validation Set"
)

plt.legend()

plt.tight_layout()


plt.savefig(

    RESULT_DIR
    / "calibration_curve.png",

    dpi=200
)

plt.close()


# ============================================================
# 15. SAVE JSON SUMMARY
# ============================================================

summary = {

    "svm_best_f1_threshold":
        float(
            svm_candidates[
                "best_f1"
            ]["threshold"]
        ),

    "svm_best_f2_threshold":
        float(
            svm_candidates[
                "best_f2"
            ]["threshold"]
        ),

    "svm_recall75_threshold":
        float(
            svm_candidates[
                "recall_at_least_075"
            ]["threshold"]
        ),

    "logistic_best_f1_threshold":
        float(
            logistic_candidates[
                "best_f1"
            ]["threshold"]
        ),
}


with open(

    RESULT_DIR
    / "threshold_summary.json",

    "w",

    encoding="utf-8"

) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 95)
print("CALIBRATION + THRESHOLD TUNING COMPLETED")
print("=" * 95)

print(
    f"\nResults:\n"
    f"{RESULT_DIR}"
)

print(
    f"\nCalibrated model:\n"
    f"{MODEL_DIR}"
)