from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse


# ============================================================
# INTERNSAFE AI
# STEP 12 - EXPLAINABLE AI
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

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake_job_postings.csv"
)

TUNED_SVM_PATH = (
    PROJECT_ROOT
    / "models"
    / "tuned"
    / "best_combined_svm.joblib"
)

CALIBRATED_PATH = (
    PROJECT_ROOT
    / "models"
    / "calibrated"
    / "calibrated_combined_svm.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "explainability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD
# ============================================================

test_df = pd.read_csv(TEST_PATH)
raw_df = pd.read_csv(RAW_PATH)

svm_pipeline = joblib.load(
    TUNED_SVM_PATH
)

calibrated_model = joblib.load(
    CALIBRATED_PATH
)


print("=" * 95)
print("INTERNSAFE AI - EXPLAINABLE AI")
print("=" * 95)


# ============================================================
# 3. FEATURE DEFINITIONS
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


# ============================================================
# 4. GET MODEL COMPONENTS
# ============================================================

preprocessor = (
    svm_pipeline
    .named_steps[
        "preprocessor"
    ]
)

classifier = (
    svm_pipeline
    .named_steps[
        "classifier"
    ]
)


feature_names = (
    preprocessor
    .get_feature_names_out()
)


coefficients = (
    classifier
    .coef_
    .ravel()
)


intercept = float(
    classifier
    .intercept_[0]
)


print(
    f"\nNumber of transformed features: "
    f"{len(feature_names):,}"
)

print(
    f"Coefficient count: "
    f"{len(coefficients):,}"
)


if len(feature_names) != len(coefficients):

    raise ValueError(
        "Feature names and model coefficients do not match."
    )


# ============================================================
# 5. MAKE FEATURE NAMES EASIER TO READ
# ============================================================

def humanize_feature_name(name):

    name = str(name)

    replacements = {

        "text__":
            "TEXT: ",

        "numeric__":
            "META: ",

        "categorical__":
            "CATEGORY: ",
    }

    for old, new in replacements.items():

        if name.startswith(old):

            name = (
                new
                + name[len(old):]
            )

            break

    return name


readable_feature_names = np.array([

    humanize_feature_name(name)

    for name in feature_names
])


# ============================================================
# 6. GLOBAL FEATURE IMPORTANCE
# ============================================================

global_features = pd.DataFrame({

    "feature":
        readable_feature_names,

    "coefficient":
        coefficients
})


global_features = (
    global_features
    .sort_values(
        "coefficient",
        ascending=False
    )
)


global_features.to_csv(

    OUTPUT_DIR
    / "01_all_global_coefficients.csv",

    index=False
)


top_fraud_features = (
    global_features
    .head(30)
)


top_real_features = (
    global_features
    .tail(30)
    .sort_values(
        "coefficient"
    )
)


top_fraud_features.to_csv(

    OUTPUT_DIR
    / "02_top_global_fraud_features.csv",

    index=False
)


top_real_features.to_csv(

    OUTPUT_DIR
    / "03_top_global_real_features.csv",

    index=False
)


print("\n" + "=" * 95)
print("TOP GLOBAL FRAUD FEATURES")
print("=" * 95)

print(
    top_fraud_features
    .head(20)
    .to_string(
        index=False
    )
)


print("\n" + "=" * 95)
print("TOP GLOBAL REAL FEATURES")
print("=" * 95)

print(
    top_real_features
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 7. GLOBAL FEATURE PLOTS
# ============================================================

plot_fraud = (
    top_fraud_features
    .head(20)
    .sort_values(
        "coefficient"
    )
)


plt.figure(
    figsize=(11, 8)
)

plt.barh(

    plot_fraud["feature"],

    plot_fraud["coefficient"]
)

plt.xlabel(
    "SVM coefficient"
)

plt.title(
    "Top Features Associated with Fraudulent Jobs"
)

plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "04_top_fraud_features.png",

    dpi=200
)

plt.close()


plot_real = (
    top_real_features
    .head(20)
    .sort_values(
        "coefficient",
        ascending=False
    )
)


plt.figure(
    figsize=(11, 8)
)

plt.barh(

    plot_real["feature"],

    plot_real["coefficient"]
)

plt.xlabel(
    "SVM coefficient"
)

plt.title(
    "Top Features Associated with Legitimate Jobs"
)

plt.tight_layout()


plt.savefig(

    OUTPUT_DIR
    / "05_top_real_features.png",

    dpi=200
)

plt.close()


# ============================================================
# 8. LOCAL EXPLANATION
# ============================================================

def explain_row(
    row,
    top_n=8
):

    row_df = pd.DataFrame(
        [row[feature_columns]]
    )


    transformed = (
        preprocessor
        .transform(
            row_df
        )
    )


    if sparse.issparse(
        transformed
    ):

        contribution = (
            transformed
            .multiply(
                coefficients
            )
            .toarray()
            .ravel()
        )

    else:

        contribution = (
            np.asarray(
                transformed
            )
            .ravel()
            * coefficients
        )


    decision_score = (
        contribution.sum()
        + intercept
    )


    fraud_probability = float(

        calibrated_model
        .predict_proba(
            row_df
        )[0, 1]
    )


    contribution_df = pd.DataFrame({

        "feature":
            readable_feature_names,

        "contribution":
            contribution
    })


    fraud_reasons = (

        contribution_df[
            contribution_df[
                "contribution"
            ] > 0
        ]

        .sort_values(
            "contribution",
            ascending=False
        )

        .head(top_n)
    )


    real_reasons = (

        contribution_df[
            contribution_df[
                "contribution"
            ] < 0
        ]

        .sort_values(
            "contribution"
        )

        .head(top_n)
    )


    return {

        "decision_score":
            decision_score,

        "fraud_probability":
            fraud_probability,

        "fraud_reasons":
            fraud_reasons,

        "real_reasons":
            real_reasons,
    }


# ============================================================
# 9. GENERATE PROBABILITIES FOR TEST
# ============================================================

test_probability = (

    calibrated_model
    .predict_proba(
        X_test
    )[:, 1]
)


test_explain = test_df.copy()

test_explain[
    "fraud_probability"
] = test_probability


# ============================================================
# 10. SELECT EXAMPLE TYPES
# ============================================================

# High-confidence true Fraud
true_fraud = (
    test_explain[
        test_explain[
            "fraudulent"
        ] == 1
    ]
    .sort_values(
        "fraud_probability",
        ascending=False
    )
)


# High-risk false positive
high_fp = (
    test_explain[
        test_explain[
            "fraudulent"
        ] == 0
    ]
    .sort_values(
        "fraud_probability",
        ascending=False
    )
)


# Missed Fraud
missed_fraud = (
    test_explain[
        test_explain[
            "fraudulent"
        ] == 1
    ]
    .sort_values(
        "fraud_probability"
    )
)


# Very safe Real
safe_real = (
    test_explain[
        test_explain[
            "fraudulent"
        ] == 0
    ]
    .sort_values(
        "fraud_probability"
    )
)


example_sets = {

    "true_fraud_high_probability":
        true_fraud.head(5),

    "false_positive_high_probability":
        high_fp.head(5),

    "missed_fraud_low_probability":
        missed_fraud.head(5),

    "true_real_low_probability":
        safe_real.head(5),
}


# ============================================================
# 11. CREATE LOCAL EXPLANATION TABLE
# ============================================================

local_rows = []


for example_type, sample in (
    example_sets.items()
):

    for _, row in (
        sample.iterrows()
    ):

        explanation = explain_row(
            row,
            top_n=6
        )


        raw_match = raw_df[
            raw_df[
                "job_id"
            ] == row["job_id"]
        ]


        title = ""

        if len(raw_match) > 0:

            title = (
                raw_match
                .iloc[0]["title"]
            )


        fraud_reason_text = " | ".join(

            explanation[
                "fraud_reasons"
            ]["feature"]
            .astype(str)
            .tolist()
        )


        real_reason_text = " | ".join(

            explanation[
                "real_reasons"
            ]["feature"]
            .astype(str)
            .tolist()
        )


        local_rows.append({

            "example_type":
                example_type,

            "job_id":
                row["job_id"],

            "title":
                title,

            "actual_fraudulent":
                row["fraudulent"],

            "fraud_probability":
                explanation[
                    "fraud_probability"
                ],

            "svm_decision_score":
                explanation[
                    "decision_score"
                ],

            "top_fraud_reasons":
                fraud_reason_text,

            "top_real_reasons":
                real_reason_text,
        })


local_explanations = pd.DataFrame(
    local_rows
)


local_explanations.to_csv(

    OUTPUT_DIR
    / "06_local_explanations.csv",

    index=False
)


# ============================================================
# 12. PRINT EXAMPLES
# ============================================================

print("\n" + "=" * 95)
print("LOCAL EXPLANATION EXAMPLES")
print("=" * 95)


for _, row in (
    local_explanations
    .head(10)
    .iterrows()
):

    print(
        "\n"
        + "-" * 95
    )

    print(
        f"Type: "
        f"{row['example_type']}"
    )

    print(
        f"Job ID: "
        f"{row['job_id']}"
    )

    print(
        f"Title: "
        f"{row['title']}"
    )

    print(
        f"Actual: "
        f"{row['actual_fraudulent']}"
    )

    print(
        f"Fraud probability: "
        f"{row['fraud_probability']:.4f}"
    )

    print(
        "\nFactors pushing toward FRAUD:"
    )

    for reason in (
        row[
            "top_fraud_reasons"
        ]
        .split(" | ")
    ):

        if reason:

            print(
                f"  + {reason}"
            )


    print(
        "\nFactors pushing toward REAL:"
    )

    for reason in (
        row[
            "top_real_reasons"
        ]
        .split(" | ")
    ):

        if reason:

            print(
                f"  - {reason}"
            )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 95)
print("EXPLAINABLE AI COMPLETED")
print("=" * 95)

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_DIR}"
)