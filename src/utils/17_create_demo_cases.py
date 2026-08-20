from pathlib import Path
import json

import pandas as pd


# ============================================================
# INTERNSAFE AI
# STEP 17 - CREATE REPRESENTATIVE DEMO CASES
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "final_test"
    / "test_predictions_with_risk.csv"
)


RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake_job_postings.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "app"
    / "demo_cases"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

pred = pd.read_csv(
    PREDICTIONS_PATH
)

raw = pd.read_csv(
    RAW_PATH
)


df = pred.merge(
    raw,
    on="job_id",
    how="left"
)


# ============================================================
# SELECT REPRESENTATIVE CASES
# ============================================================

# HIGH:
# Fraud case with very high probability
high_case = (
    df[
        (df["risk_level"] == "HIGH")
        &
        (df["actual_fraudulent"] == 1)
    ]
    .sort_values(
        "fraud_probability",
        ascending=False
    )
    .iloc[0]
)


# REVIEW:
# Pick example nearest 30% probability
review_pool = df[
    df["risk_level"] == "REVIEW"
].copy()

review_pool[
    "distance"
] = abs(
    review_pool[
        "fraud_probability"
    ] - 0.30
)

review_case = (
    review_pool
    .sort_values(
        "distance"
    )
    .iloc[0]
)


# LOW:
# Legitimate case with extremely low probability
low_case = (
    df[
        (df["risk_level"] == "LOW")
        &
        (df["actual_fraudulent"] == 0)
    ]
    .sort_values(
        "fraud_probability"
    )
    .iloc[0]
)


# ============================================================
# CONVERT RAW ROW -> API INPUT
# ============================================================

FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
]


def clean_value(value):

    if pd.isna(value):
        return ""

    return value


def make_job(row):

    job = {}

    for field in FIELDS:

        value = clean_value(
            row[field]
        )

        if field in [
            "telecommuting",
            "has_company_logo",
            "has_questions",
        ]:

            value = int(
                value
                if value != ""
                else 0
            )

        job[field] = value

    return job


# ============================================================
# SAVE
# ============================================================

cases = {

    "LOW": {
        "expected_level":
            "LOW",

        "expected_probability":
            float(
                low_case[
                    "fraud_probability"
                ]
            ),

        "actual_fraudulent":
            int(
                low_case[
                    "actual_fraudulent"
                ]
            ),

        "job":
            make_job(
                low_case
            )
    },


    "REVIEW": {
        "expected_level":
            "REVIEW",

        "expected_probability":
            float(
                review_case[
                    "fraud_probability"
                ]
            ),

        "actual_fraudulent":
            int(
                review_case[
                    "actual_fraudulent"
                ]
            ),

        "job":
            make_job(
                review_case
            )
    },


    "HIGH": {
        "expected_level":
            "HIGH",

        "expected_probability":
            float(
                high_case[
                    "fraud_probability"
                ]
            ),

        "actual_fraudulent":
            int(
                high_case[
                    "actual_fraudulent"
                ]
            ),

        "job":
            make_job(
                high_case
            )
    },
}


OUTPUT_PATH = (
    OUTPUT_DIR
    / "demo_cases.json"
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        cases,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# DISPLAY
# ============================================================

print("=" * 90)
print("INTERNSAFE AI - DEMO CASES")
print("=" * 90)


for level, case in cases.items():

    print(
        f"\n{level}"
    )

    print(
        "-" * 60
    )

    print(
        "Title:",
        case[
            "job"
        ][
            "title"
        ]
    )

    print(
        "Expected probability:",
        round(
            case[
                "expected_probability"
            ] * 100,
            2
        ),
        "%"
    )

    print(
        "Expected risk level:",
        case[
            "expected_level"
        ]
    )

    print(
        "Actual label:",
        case[
            "actual_fraudulent"
        ]
    )


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)