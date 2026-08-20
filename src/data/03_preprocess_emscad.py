from pathlib import Path
import hashlib
import html
import re

import pandas as pd


# ============================================================
# INTERNSAFE AI
# STEP 03 - DETERMINISTIC PREPROCESSING
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake_job_postings.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "emscad_preprocessed_base.csv"
)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("=" * 85)
print("INTERNSAFE AI - PREPROCESSING EMSCAD")
print("=" * 85)

print(f"\nInput shape: {df.shape}")


# ============================================================
# 1. TEXT HELPERS
# ============================================================

def clean_text(text):
    """
    Clean obvious HTML/noise while preserving useful
    fraud-related information through special tokens.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Replace URLs with a token
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URLTOKEN ",
        text,
        flags=re.IGNORECASE
    )

    # Replace email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}\b",
        " EMAILTOKEN ",
        text
    )

    # Replace phone-like numbers
    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " PHONETOKEN ",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_for_group(text):
    """
    Normalize content only for duplicate-group detection.

    This is NOT the text used by the ML model.
    """

    if pd.isna(text):
        return ""

    text = html.unescape(str(text))

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def make_group_hash(text, job_id):
    """
    Create deterministic group ID so duplicate/normalized
    duplicate postings stay in the same data split.
    """

    text = str(text).strip()

    if not text:
        text = f"EMPTY_JOB_{job_id}"

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# 2. RAW TEXT SIGNALS
# ============================================================

raw_text = (
    df["title"].fillna("").astype(str)
    + " "
    + df["company_profile"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
    + " "
    + df["requirements"].fillna("").astype(str)
    + " "
    + df["benefits"].fillna("").astype(str)
)


df["has_url"] = (
    raw_text
    .str.contains(
        r"https?://|www\.",
        case=False,
        regex=True
    )
    .astype(int)
)


df["has_email"] = (
    raw_text
    .str.contains(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}\b",
        regex=True
    )
    .astype(int)
)


df["has_phone_like"] = (
    raw_text
    .str.contains(
        r"\+?\d[\d\s().-]{7,}\d",
        regex=True
    )
    .astype(int)
)


df["has_html"] = (
    raw_text
    .str.contains(
        r"<[^>]+>",
        regex=True
    )
    .astype(int)
)


df["has_currency_symbol"] = (
    raw_text
    .str.contains(
        r"[$€£]",
        regex=True
    )
    .astype(int)
)


df["exclamation_count"] = (
    raw_text
    .str.count("!")
)


def count_all_caps_words(text):

    return len(
        re.findall(
            r"\b[A-Z]{3,}\b",
            str(text)
        )
    )


df["all_caps_word_count"] = (
    raw_text.map(
        count_all_caps_words
    )
)


# ============================================================
# 3. MISSINGNESS FEATURES
# ============================================================

missing_sources = [
    "company_profile",
    "requirements",
    "benefits",
    "salary_range",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]


for column in missing_sources:

    df[f"missing_{column}"] = (
        df[column]
        .isna()
        .astype(int)
    )


# Friendly salary feature
df["has_salary"] = (
    (
        df["salary_range"]
        .notna()
    )
    &
    (
        df["salary_range"]
        .fillna("")
        .astype(str)
        .str.strip()
        != ""
    )
).astype(int)


# ============================================================
# 4. CLEAN EACH TEXT FIELD
# ============================================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]


for column in text_columns:

    clean_column = f"{column}_clean"

    df[clean_column] = (
        df[column]
        .apply(clean_text)
    )


# ============================================================
# 5. TEXT LENGTH FEATURES
# ============================================================

for column in text_columns:

    cleaned = df[f"{column}_clean"]

    df[f"{column}_char_length"] = (
        cleaned
        .str.len()
    )

    df[f"{column}_word_length"] = (
        cleaned
        .str.split()
        .str.len()
    )


# ============================================================
# 6. COMBINE TEXT
# ============================================================

df["combined_text"] = (
    "TITLE "
    + df["title_clean"]
    + " COMPANY "
    + df["company_profile_clean"]
    + " DESCRIPTION "
    + df["description_clean"]
    + " REQUIREMENTS "
    + df["requirements_clean"]
    + " BENEFITS "
    + df["benefits_clean"]
)


# ============================================================
# 7. COUNTRY FROM LOCATION
# ============================================================

df["country"] = (
    df["location"]
    .fillna("")
    .astype(str)
    .str.split(",")
    .str[0]
    .str.strip()
)


df.loc[
    df["country"] == "",
    "country"
] = "Missing"


# ============================================================
# 8. FIXED MISSING TOKENS FOR CATEGORICAL DATA
# ============================================================

categorical_columns = [
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]


for column in categorical_columns:

    df[column] = (
        df[column]
        .fillna("Missing")
        .astype(str)
        .str.strip()
    )

    df.loc[
        df[column] == "",
        column
    ] = "Missing"


# ============================================================
# 9. DUPLICATE GROUP IDENTIFIER
# ============================================================

group_content = (
    df["title"]
    .fillna("")
    .astype(str)
    + " "
    + df["description"]
    .fillna("")
    .astype(str)
)


df["_group_normalized_text"] = (
    group_content
    .map(normalize_for_group)
)


df["content_group"] = [
    make_group_hash(text, job_id)
    for text, job_id in zip(
        df["_group_normalized_text"],
        df["job_id"]
    )
]


# ============================================================
# 10. SELECT OUTPUT COLUMNS
# ============================================================

output_columns = [
    # Traceability
    "job_id",

    # Used later for leakage-safe splitting
    "content_group",

    # Target
    "fraudulent",

    # Main NLP input
    "combined_text",

    # Individual text if later experiments need them
    "title_clean",
    "company_profile_clean",
    "description_clean",
    "requirements_clean",
    "benefits_clean",

    # Binary metadata from original dataset
    "telecommuting",
    "has_company_logo",
    "has_questions",

    # Missingness information
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

    # Lower-cardinality categorical metadata
    "country",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",

    # Keep department for later experiment,
    # but do not automatically use in baseline
    "department"
]


processed = df[
    output_columns
].copy()


# ============================================================
# 11. VALIDATION
# ============================================================

print("\n" + "=" * 85)
print("PREPROCESSING VALIDATION")
print("=" * 85)


print(
    f"Output rows       : "
    f"{len(processed):,}"
)

print(
    f"Output columns    : "
    f"{len(processed.columns):,}"
)

print(
    f"Unique job IDs    : "
    f"{processed['job_id'].nunique():,}"
)

print(
    f"Unique groups     : "
    f"{processed['content_group'].nunique():,}"
)

print(
    f"Duplicate-group rows: "
    f"{len(processed) - processed['content_group'].nunique():,}"
)


print("\nTarget distribution:")

print(
    processed["fraudulent"]
    .value_counts()
    .sort_index()
)


# Verify each content group has only one label
group_label_counts = (
    processed
    .groupby("content_group")
    ["fraudulent"]
    .nunique()
)


conflicting_groups = (
    group_label_counts > 1
).sum()


print(
    f"\nContent groups with conflicting labels: "
    f"{conflicting_groups}"
)


if conflicting_groups != 0:

    raise ValueError(
        "Có duplicate groups chứa nhiều label khác nhau."
    )


# Check text completeness
empty_combined = (
    processed["combined_text"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

print(
    f"Empty combined_text: "
    f"{empty_combined}"
)


# ============================================================
# 12. SAVE
# ============================================================

processed.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 85)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 85)

print(
    f"\nSaved dataset:\n{OUTPUT_PATH}"
)