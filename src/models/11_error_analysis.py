from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INTERNSAFE AI
# STEP 11 - FINAL MODEL ERROR ANALYSIS
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake_job_postings.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_test"
    / "test_predictions_with_risk.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

raw_df = pd.read_csv(RAW_PATH)

pred_df = pd.read_csv(
    PREDICTION_PATH
)


df = pred_df.merge(
    raw_df,
    on="job_id",
    how="left"
)


print("=" * 95)
print("INTERNSAFE AI - ERROR ANALYSIS")
print("=" * 95)

print(
    f"\nTest predictions: "
    f"{len(df):,}"
)


# ============================================================
# 3. FIXED THRESHOLDS
# ============================================================

SCREENING_THRESHOLD = 0.120
STRICT_THRESHOLD = 0.520


df["screening_prediction"] = (
    df["fraud_probability"]
    >= SCREENING_THRESHOLD
).astype(int)


df["strict_prediction"] = (
    df["fraud_probability"]
    >= STRICT_THRESHOLD
).astype(int)


# ============================================================
# 4. SCREENING ERRORS
# ============================================================

screening_fp = df[
    (df["actual_fraudulent"] == 0)
    &
    (df["screening_prediction"] == 1)
].copy()


screening_fn = df[
    (df["actual_fraudulent"] == 1)
    &
    (df["screening_prediction"] == 0)
].copy()


# ============================================================
# 5. STRICT ERRORS
# ============================================================

strict_fp = df[
    (df["actual_fraudulent"] == 0)
    &
    (df["strict_prediction"] == 1)
].copy()


strict_fn = df[
    (df["actual_fraudulent"] == 1)
    &
    (df["strict_prediction"] == 0)
].copy()


# ============================================================
# 6. ERROR SUMMARY
# ============================================================

summary = pd.DataFrame({

    "error_type": [
        "Screening False Positive",
        "Screening False Negative",
        "Strict False Positive",
        "Strict False Negative"
    ],

    "count": [
        len(screening_fp),
        len(screening_fn),
        len(strict_fp),
        len(strict_fn)
    ]
})


print("\n" + "=" * 95)
print("ERROR SUMMARY")
print("=" * 95)

print(
    summary.to_string(
        index=False
    )
)


summary.to_csv(
    OUTPUT_DIR
    / "01_error_summary.csv",
    index=False
)


# ============================================================
# 7. COLUMNS FOR MANUAL ANALYSIS
# ============================================================

analysis_columns = [

    "job_id",

    "fraud_probability",
    "risk_level",

    "title",
    "location",

    "company_profile",
    "description",
    "requirements",
    "benefits",

    "telecommuting",
    "has_company_logo",
    "has_questions",

    "employment_type",
    "required_experience",
    "required_education",

    "industry",
    "function",

    "salary_range",

    "actual_fraudulent"
]


# ============================================================
# 8. SAVE ERROR RECORDS
# ============================================================

screening_fp[
    analysis_columns
].sort_values(
    "fraud_probability",
    ascending=False
).to_csv(

    OUTPUT_DIR
    / "02_screening_false_positives.csv",

    index=False
)


screening_fn[
    analysis_columns
].sort_values(
    "fraud_probability"
).to_csv(

    OUTPUT_DIR
    / "03_screening_false_negatives.csv",

    index=False
)


strict_fp[
    analysis_columns
].sort_values(
    "fraud_probability",
    ascending=False
).to_csv(

    OUTPUT_DIR
    / "04_strict_false_positives.csv",

    index=False
)


strict_fn[
    analysis_columns
].sort_values(
    "fraud_probability"
).to_csv(

    OUTPUT_DIR
    / "05_strict_false_negatives.csv",

    index=False
)


# ============================================================
# 9. MISSED FRAUD ANALYSIS
# ============================================================

missed_fraud = screening_fn.copy()


print("\n" + "=" * 95)
print("MISSED FRAUD AT SAFETY SCREENING")
print("=" * 95)

print(
    f"\nTotal missed Fraud: "
    f"{len(missed_fraud)}"
)


if len(missed_fraud) > 0:

    print(
        "\nAverage probability:"
    )

    print(
        round(
            missed_fraud[
                "fraud_probability"
            ].mean(),
            4
        )
    )


    print(
        "\nHas company logo:"
    )

    print(
        missed_fraud[
            "has_company_logo"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\nHas questions:"
    )

    print(
        missed_fraud[
            "has_questions"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\nEmployment type:"
    )

    print(
        missed_fraud[
            "employment_type"
        ]
        .value_counts(
            dropna=False
        )
    )


# ============================================================
# 10. HIGH-RISK FALSE POSITIVES
# ============================================================

print("\n" + "=" * 95)
print("FALSE ALARMS IN HIGH-RISK GROUP")
print("=" * 95)

print(
    f"\nHigh-risk false positives: "
    f"{len(strict_fp)}"
)


if len(strict_fp) > 0:

    print(
        "\nTitles:"
    )

    for _, row in (
        strict_fp
        .sort_values(
            "fraud_probability",
            ascending=False
        )
        .iterrows()
    ):

        print(
            f"\n"
            f"[{row['fraud_probability']:.3f}] "
            f"{row['title']}"
        )


# ============================================================
# 11. RISK-LEVEL BREAKDOWN
# ============================================================

risk_summary = (

    df
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

        mean_probability=(
            "fraud_probability",
            "mean"
        )
    )
)


risk_summary[
    "fraud_rate"
] = (

    risk_summary[
        "fraud_jobs"
    ]

    /

    risk_summary[
        "jobs"
    ]
)


risk_summary.to_csv(

    OUTPUT_DIR
    / "06_risk_level_analysis.csv"
)


# ============================================================
# 12. PROBABILITY DISTRIBUTION
# ============================================================

real_probability = df.loc[

    df[
        "actual_fraudulent"
    ] == 0,

    "fraud_probability"
]


fraud_probability = df.loc[

    df[
        "actual_fraudulent"
    ] == 1,

    "fraud_probability"
]


plt.figure(
    figsize=(9, 6)
)


plt.hist(

    real_probability,

    bins=40,

    alpha=0.6,

    label="Real"
)


plt.hist(

    fraud_probability,

    bins=40,

    alpha=0.6,

    label="Fraudulent"
)


plt.axvline(

    SCREENING_THRESHOLD,

    linestyle="--",

    label="Screening threshold"
)


plt.axvline(

    STRICT_THRESHOLD,

    linestyle="--",

    label="High-risk threshold"
)


plt.xlabel(
    "Predicted Fraud Probability"
)

plt.ylabel(
    "Number of Jobs"
)

plt.title(
    "Fraud Probability Distribution on Final Test Set"
)

plt.legend()

plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "07_probability_distribution.png",

    dpi=200
)

plt.close()


# ============================================================
# 13. SAVE ALL FALSE NEGATIVES WITH SHORT TEXT
# ============================================================

short_error_view = (

    screening_fn[

        [
            "job_id",
            "fraud_probability",
            "title",
            "location",
            "employment_type",
            "industry",
            "has_company_logo",
            "has_questions",
            "salary_range"
        ]
    ]
    .sort_values(
        "fraud_probability"
    )
)


short_error_view.to_csv(

    OUTPUT_DIR
    / "08_missed_fraud_short_view.csv",

    index=False
)


print("\n" + "=" * 95)
print("ERROR ANALYSIS COMPLETED")
print("=" * 95)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_DIR}"
)