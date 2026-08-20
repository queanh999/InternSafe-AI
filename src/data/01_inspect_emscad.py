from pathlib import Path

import pandas as pd


# ==========================================================
# INTERNSAFE AI
# STEP 01 - INSPECT EMSCAD DATASET
# ==========================================================


# ----------------------------------------------------------
# PATH
# ----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake_job_postings.csv"
)


print("=" * 80)
print("INTERNSAFE AI - EMSCAD DATA INSPECTION")
print("=" * 80)

print("\nDataset path:")
print(DATA_PATH)


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy dataset tại:\n{DATA_PATH}"
    )


# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ==========================================================
# 1. SIZE
# ==========================================================

print("\n" + "=" * 80)
print("1. DATASET SIZE")
print("=" * 80)

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}")


# ==========================================================
# 2. COLUMN NAMES
# ==========================================================

print("\n" + "=" * 80)
print("2. COLUMN NAMES")
print("=" * 80)

for i, column in enumerate(
    df.columns,
    start=1
):
    print(f"{i:02d}. {column}")


# ==========================================================
# 3. DATA TYPES
# ==========================================================

print("\n" + "=" * 80)
print("3. DATA TYPES")
print("=" * 80)

print(df.dtypes)


# ==========================================================
# 4. MISSING VALUES
# ==========================================================

print("\n" + "=" * 80)
print("4. MISSING VALUES")
print("=" * 80)

missing = pd.DataFrame({
    "missing_count":
        df.isnull().sum(),

    "missing_percent":
        (
            df.isnull().mean()
            * 100
        ).round(2)
})

missing = (
    missing
    .sort_values(
        "missing_percent",
        ascending=False
    )
)

print(missing)


# ==========================================================
# 5. DUPLICATE ROWS
# ==========================================================

print("\n" + "=" * 80)
print("5. DUPLICATES")
print("=" * 80)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


if "job_id" in df.columns:

    print(
        "Duplicate job_id:",
        df["job_id"]
        .duplicated()
        .sum()
    )


# ==========================================================
# 6. TARGET DISTRIBUTION
# ==========================================================

print("\n" + "=" * 80)
print("6. TARGET DISTRIBUTION")
print("=" * 80)


if "fraudulent" not in df.columns:

    raise ValueError(
        "Không tìm thấy target 'fraudulent'."
    )


target_counts = (
    df["fraudulent"]
    .value_counts()
    .sort_index()
)

target_percent = (
    df["fraudulent"]
    .value_counts(
        normalize=True
    )
    .sort_index()
    .mul(100)
    .round(2)
)


target_summary = pd.DataFrame({
    "count": target_counts,
    "percent": target_percent
})


print(target_summary)


# ==========================================================
# 7. TARGET LABEL EXPLANATION
# ==========================================================

print("\nTarget meaning:")

print("0 = Real / Legitimate job")
print("1 = Fraudulent job")


# ==========================================================
# 8. FRAUD RATE
# ==========================================================

fraud_rate = (
    df["fraudulent"]
    .mean()
    * 100
)

print(
    f"\nFraud rate: "
    f"{fraud_rate:.2f}%"
)


# ==========================================================
# 9. TEXT INFORMATION
# ==========================================================

print("\n" + "=" * 80)
print("7. TEXT INFORMATION")
print("=" * 80)


text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]


for column in text_columns:

    if column in df.columns:

        text = (
            df[column]
            .fillna("")
            .astype(str)
        )

        average_chars = (
            text
            .str.len()
            .mean()
        )

        average_words = (
            text
            .str.split()
            .str.len()
            .mean()
        )

        print(
            f"\n{column}:"
        )

        print(
            f"  Average characters: "
            f"{average_chars:.2f}"
        )

        print(
            f"  Average words: "
            f"{average_words:.2f}"
        )


# ==========================================================
# 10. CATEGORICAL FEATURES
# ==========================================================

print("\n" + "=" * 80)
print("8. IMPORTANT CATEGORICAL FEATURES")
print("=" * 80)


categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]


for column in categorical_columns:

    if column in df.columns:

        print(
            f"\n--- {column} ---"
        )

        print(
            df[column]
            .value_counts(
                dropna=False
            )
            .head(10)
        )


# ==========================================================
# 11. BINARY FEATURES
# ==========================================================

print("\n" + "=" * 80)
print("9. BINARY FEATURES")
print("=" * 80)


binary_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]


for column in binary_columns:

    if column in df.columns:

        print(
            f"\n--- {column} ---"
        )

        print(
            df[column]
            .value_counts(
                dropna=False
            )
        )


# ==========================================================
# 12. SAMPLE RECORDS
# ==========================================================

print("\n" + "=" * 80)
print("10. SAMPLE RECORDS")
print("=" * 80)


sample_columns = [
    column
    for column in [
        "job_id",
        "title",
        "location",
        "employment_type",
        "fraudulent"
    ]
    if column in df.columns
]


pd.set_option(
    "display.max_colwidth",
    100
)


print(
    df[sample_columns]
    .head(10)
    .to_string(
        index=False
    )
)


# ==========================================================
# FINISH
# ==========================================================

print("\n" + "=" * 80)
print("EMSCAD DATA INSPECTION COMPLETED")
print("=" * 80)