from pathlib import Path
import html
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# INTERNSAFE AI
# COMPLETE EXPLORATORY DATA ANALYSIS
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake_job_postings.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "eda_complete"
)

FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
SAMPLE_DIR = OUTPUT_ROOT / "manual_samples"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 85)
print("INTERNSAFE AI - COMPLETE EMSCAD EDA")
print("=" * 85)

print(f"\nDataset shape : {df.shape}")
print(f"Total jobs    : {len(df):,}")
print(f"Real jobs     : {(df['fraudulent'] == 0).sum():,}")
print(f"Fraud jobs    : {(df['fraudulent'] == 1).sum():,}")
print(
    f"Fraud rate    : "
    f"{df['fraudulent'].mean() * 100:.2f}%"
)


# ============================================================
# 3. BASIC OVERVIEW
# ============================================================

overview = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "real_jobs",
        "fraud_jobs",
        "fraud_rate_percent"
    ],
    "value": [
        len(df),
        len(df.columns),
        int((df["fraudulent"] == 0).sum()),
        int((df["fraudulent"] == 1).sum()),
        round(df["fraudulent"].mean() * 100, 2)
    ]
})

overview.to_csv(
    TABLE_DIR / "00_dataset_overview.csv",
    index=False
)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

target_counts = (
    df["fraudulent"]
    .value_counts()
    .sort_index()
)

target_summary = pd.DataFrame({
    "count": target_counts,
    "percent": (
        target_counts
        / len(df)
        * 100
    ).round(2)
})

target_summary.index = [
    "Real",
    "Fraudulent"
]

target_summary.to_csv(
    TABLE_DIR / "01_target_distribution.csv"
)

plt.figure(figsize=(7, 5))

plt.bar(
    ["Real", "Fraudulent"],
    target_counts.values
)

plt.title(
    "Distribution of Real and Fraudulent Job Postings"
)

plt.ylabel("Number of postings")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_target_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# 5. MISSING VALUES - OVERALL
# ============================================================

original_columns = list(df.columns)

missing_overall = pd.DataFrame({
    "missing_count":
        df[original_columns].isna().sum(),

    "missing_percent":
        (
            df[original_columns]
            .isna()
            .mean()
            * 100
        ).round(2)
})

missing_overall = (
    missing_overall
    .sort_values(
        "missing_percent",
        ascending=False
    )
)

missing_overall.to_csv(
    TABLE_DIR / "02_missing_overall.csv"
)


missing_plot = missing_overall[
    missing_overall["missing_percent"] > 0
]

plt.figure(figsize=(11, 6))

plt.bar(
    missing_plot.index,
    missing_plot["missing_percent"]
)

plt.title("Missing Values in EMSCAD")
plt.xlabel("Feature")
plt.ylabel("Missing (%)")

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_missing_overall.png",
    dpi=200
)

plt.close()


# ============================================================
# 6. MISSING VALUES: REAL VS FRAUD
# ============================================================

real_df = df[df["fraudulent"] == 0]
fraud_df = df[df["fraudulent"] == 1]

missing_by_class = pd.DataFrame({
    "real_missing_percent":
        (
            real_df[original_columns]
            .isna()
            .mean()
            * 100
        ),

    "fraud_missing_percent":
        (
            fraud_df[original_columns]
            .isna()
            .mean()
            * 100
        )
})

missing_by_class["gap_fraud_minus_real"] = (
    missing_by_class["fraud_missing_percent"]
    - missing_by_class["real_missing_percent"]
)

missing_by_class = missing_by_class.round(2)

missing_by_class["absolute_gap"] = (
    missing_by_class[
        "gap_fraud_minus_real"
    ].abs()
)

missing_by_class = (
    missing_by_class
    .sort_values(
        "absolute_gap",
        ascending=False
    )
)

missing_by_class.to_csv(
    TABLE_DIR / "03_missing_real_vs_fraud.csv"
)


missing_gap_plot = (
    missing_by_class
    .drop(
        index="fraudulent",
        errors="ignore"
    )
    .head(12)
)

plt.figure(figsize=(10, 6))

plt.bar(
    missing_gap_plot.index,
    missing_gap_plot[
        "gap_fraud_minus_real"
    ]
)

plt.title(
    "Difference in Missingness: Fraud vs Real"
)

plt.xlabel("Feature")

plt.ylabel(
    "Fraud missing % - Real missing %"
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_missing_gap_real_vs_fraud.png",
    dpi=200
)

plt.close()


# ============================================================
# 7. DUPLICATE / LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 85)
print("DUPLICATE AND LEAKAGE CHECK")
print("=" * 85)


# Ignore job_id because it is unique by design
duplicate_feature_columns = [
    col
    for col in original_columns
    if col != "job_id"
]

duplicate_without_id = (
    df.duplicated(
        subset=duplicate_feature_columns
    )
    .sum()
)


# ------------------------------------------------------------
# Exact duplicate descriptions
# ------------------------------------------------------------

description_nonempty = (
    df["description"]
    .dropna()
    .astype(str)
    .str.strip()
)

description_nonempty = (
    description_nonempty[
        description_nonempty != ""
    ]
)

duplicate_descriptions = (
    description_nonempty
    .duplicated()
    .sum()
)


# ------------------------------------------------------------
# Exact title + description duplicates
# ------------------------------------------------------------

combined_exact = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
    + " || "
    +
    df["description"]
    .fillna("")
    .astype(str)
    .str.strip()
)

combined_nonempty = combined_exact[
    combined_exact.str.replace(
        "||",
        "",
        regex=False
    ).str.strip() != ""
]

duplicate_combined_exact = (
    combined_nonempty
    .duplicated()
    .sum()
)


# ------------------------------------------------------------
# Normalize for near-exact duplicate detection
# ------------------------------------------------------------

def normalize_for_duplicate(text):

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


df["_normalized_content"] = (
    df["title"]
    .fillna("")
    .astype(str)
    + " "
    +
    df["description"]
    .fillna("")
    .astype(str)
)

df["_normalized_content"] = (
    df["_normalized_content"]
    .map(normalize_for_duplicate)
)


normalized_nonempty = df[
    df["_normalized_content"] != ""
].copy()


duplicate_normalized = (
    normalized_nonempty[
        "_normalized_content"
    ]
    .duplicated()
    .sum()
)


# ------------------------------------------------------------
# Check conflicting labels
# ------------------------------------------------------------

duplicate_groups = (
    normalized_nonempty
    .groupby("_normalized_content")
    .agg(
        records=(
            "job_id",
            "count"
        ),
        different_labels=(
            "fraudulent",
            "nunique"
        )
    )
)

duplicate_groups = (
    duplicate_groups[
        duplicate_groups["records"] > 1
    ]
)

conflicting_groups = (
    duplicate_groups[
        duplicate_groups[
            "different_labels"
        ] > 1
    ]
)


duplicate_summary = pd.DataFrame({
    "check": [
        "duplicate_rows_excluding_job_id",
        "duplicate_descriptions",
        "duplicate_exact_title_description",
        "duplicate_normalized_content",
        "duplicate_normalized_groups",
        "conflicting_label_groups"
    ],

    "count": [
        duplicate_without_id,
        duplicate_descriptions,
        duplicate_combined_exact,
        duplicate_normalized,
        len(duplicate_groups),
        len(conflicting_groups)
    ]
})

duplicate_summary.to_csv(
    TABLE_DIR / "04_duplicate_leakage_summary.csv",
    index=False
)

print(duplicate_summary.to_string(index=False))


# Save duplicate groups
if len(duplicate_groups) > 0:

    duplicate_groups.reset_index().to_csv(
        TABLE_DIR / "05_duplicate_groups.csv",
        index=False
    )


# ============================================================
# 8. TEXT LENGTH FEATURES
# ============================================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:

    temp_text = (
        df[column]
        .fillna("")
        .astype(str)
    )

    df[f"{column}_char_length"] = (
        temp_text
        .str.len()
    )

    df[f"{column}_word_length"] = (
        temp_text
        .str.split()
        .str.len()
    )


# ============================================================
# 9. TEXT STATISTICS BY CLASS
# ============================================================

text_statistics_rows = []

for column in text_columns:

    feature = f"{column}_word_length"

    for label_value, label_name in [
        (0, "Real"),
        (1, "Fraudulent")
    ]:

        values = df.loc[
            df["fraudulent"] == label_value,
            feature
        ]

        text_statistics_rows.append({
            "text_field": column,
            "class": label_name,
            "mean_words":
                round(values.mean(), 2),

            "median_words":
                round(values.median(), 2),

            "q1_words":
                round(values.quantile(0.25), 2),

            "q3_words":
                round(values.quantile(0.75), 2),

            "p95_words":
                round(values.quantile(0.95), 2),

            "max_words":
                int(values.max())
        })


text_statistics = pd.DataFrame(
    text_statistics_rows
)

text_statistics.to_csv(
    TABLE_DIR / "06_text_statistics_by_class.csv",
    index=False
)


print("\n" + "=" * 85)
print("TEXT LENGTH STATISTICS")
print("=" * 85)

print(
    text_statistics.to_string(
        index=False
    )
)


# ============================================================
# 10. DESCRIPTION LENGTH BOXPLOT
# ============================================================

real_description = df.loc[
    df["fraudulent"] == 0,
    "description_word_length"
]

fraud_description = df.loc[
    df["fraudulent"] == 1,
    "description_word_length"
]


# Clip only for visualization
upper_limit = (
    df["description_word_length"]
    .quantile(0.99)
)

real_plot = real_description[
    real_description <= upper_limit
]

fraud_plot = fraud_description[
    fraud_description <= upper_limit
]


plt.figure(figsize=(7, 6))

plt.boxplot(
    [
        real_plot,
        fraud_plot
    ],
    tick_labels=[
        "Real",
        "Fraudulent"
    ]
)

plt.title(
    "Description Length by Job Class"
)

plt.ylabel(
    "Description length (words)"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_description_length_boxplot.png",
    dpi=200
)

plt.close()


# ============================================================
# 11. CREATE MISSINGNESS FEATURES FOR ANALYSIS
# ============================================================

missing_features = {
    "missing_company_profile":
        "company_profile",

    "missing_requirements":
        "requirements",

    "missing_benefits":
        "benefits",

    "missing_salary":
        "salary_range",

    "missing_department":
        "department",

    "missing_education":
        "required_education",

    "missing_experience":
        "required_experience",

    "missing_industry":
        "industry",

    "missing_function":
        "function"
}


for new_feature, source_column in (
    missing_features.items()
):

    df[new_feature] = (
        df[source_column]
        .isna()
        .astype(int)
    )


# ============================================================
# 12. BINARY FEATURES VS FRAUD
# ============================================================

binary_features = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
] + list(missing_features.keys())


binary_rows = []

for feature in binary_features:

    temp = (
        df.groupby(feature)
        ["fraudulent"]
        .agg(
            count="count",
            fraud_count="sum",
            fraud_rate="mean"
        )
        .reset_index()
    )

    temp["fraud_rate"] *= 100

    temp["feature"] = feature

    binary_rows.append(temp)


binary_summary = pd.concat(
    binary_rows,
    ignore_index=True
)

binary_summary.to_csv(
    TABLE_DIR / "07_binary_feature_fraud_rates.csv",
    index=False
)


# Selected features for readable plot
selected_binary = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "missing_company_profile",
    "missing_requirements",
    "missing_salary"
]

plot_rows = []

for feature in selected_binary:

    temp = binary_summary[
        binary_summary["feature"] == feature
    ]

    for _, row in temp.iterrows():

        plot_rows.append({
            "label":
                f"{feature}={int(row[feature])}",

            "fraud_rate":
                row["fraud_rate"]
        })


binary_plot = pd.DataFrame(
    plot_rows
)

plt.figure(figsize=(12, 6))

plt.bar(
    binary_plot["label"],
    binary_plot["fraud_rate"]
)

plt.title(
    "Fraud Rate by Binary and Missingness Features"
)

plt.ylabel("Fraud rate (%)")

plt.xticks(
    rotation=55,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_binary_feature_fraud_rate.png",
    dpi=200
)

plt.close()


# ============================================================
# 13. TEXT / CONTACT / FORMATTING SIGNALS
# ============================================================

combined_text = (
    df["title"]
    .fillna("")
    .astype(str)
    + " "
    +
    df["company_profile"]
    .fillna("")
    .astype(str)
    + " "
    +
    df["description"]
    .fillna("")
    .astype(str)
    + " "
    +
    df["requirements"]
    .fillna("")
    .astype(str)
    + " "
    +
    df["benefits"]
    .fillna("")
    .astype(str)
)


df["has_url"] = (
    combined_text
    .str.contains(
        r"https?://|www\.",
        case=False,
        regex=True
    )
    .astype(int)
)


df["has_email"] = (
    combined_text
    .str.contains(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}\b",
        regex=True
    )
    .astype(int)
)


df["has_phone_like"] = (
    combined_text
    .str.contains(
        r"\+?\d[\d\s().-]{7,}\d",
        regex=True
    )
    .astype(int)
)


df["has_html"] = (
    combined_text
    .str.contains(
        r"<[^>]+>",
        regex=True
    )
    .astype(int)
)


df["has_currency_symbol"] = (
    combined_text
    .str.contains(
        r"[$€£]",
        regex=True
    )
    .astype(int)
)


df["exclamation_count"] = (
    combined_text
    .str.count("!")
)


def count_all_caps_words(text):

    words = re.findall(
        r"\b[A-Z]{3,}\b",
        str(text)
    )

    return len(words)


df["all_caps_word_count"] = (
    combined_text
    .map(count_all_caps_words)
)


text_signal_features = [
    "has_url",
    "has_email",
    "has_phone_like",
    "has_html",
    "has_currency_symbol"
]


text_signal_rows = []

for feature in text_signal_features:

    temp = (
        df.groupby(feature)
        ["fraudulent"]
        .agg(
            count="count",
            fraud_count="sum",
            fraud_rate="mean"
        )
        .reset_index()
    )

    temp["fraud_rate"] *= 100
    temp["feature"] = feature

    text_signal_rows.append(temp)


text_signal_summary = pd.concat(
    text_signal_rows,
    ignore_index=True
)

text_signal_summary.to_csv(
    TABLE_DIR / "08_text_signal_fraud_rates.csv",
    index=False
)


signal_positive = (
    text_signal_summary[
        text_signal_summary[
            text_signal_summary.columns[0]
        ].notna()
    ]
)


plot_signal_rows = []

for feature in text_signal_features:

    row = text_signal_summary[
        (text_signal_summary["feature"] == feature)
        &
        (
            text_signal_summary[feature] == 1
        )
    ]

    if not row.empty:

        plot_signal_rows.append({
            "feature": feature,
            "fraud_rate":
                row["fraud_rate"].iloc[0]
        })


signal_plot = pd.DataFrame(
    plot_signal_rows
)

plt.figure(figsize=(9, 5))

plt.bar(
    signal_plot["feature"],
    signal_plot["fraud_rate"]
)

plt.title(
    "Fraud Rate When Text Signal Is Present"
)

plt.ylabel("Fraud rate (%)")

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_text_signal_fraud_rate.png",
    dpi=200
)

plt.close()


# Continuous formatting signal summary
format_summary = (
    df.groupby("fraudulent")[
        [
            "exclamation_count",
            "all_caps_word_count"
        ]
    ]
    .agg(
        ["mean", "median", "max"]
    )
)

format_summary.to_csv(
    TABLE_DIR / "09_formatting_signal_summary.csv"
)


# ============================================================
# 14. CARDINALITY CHECK
# ============================================================

categorical_columns = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]


cardinality_rows = []

for column in categorical_columns:

    value_counts = (
        df[column]
        .fillna("Missing")
        .value_counts()
    )

    cardinality_rows.append({
        "feature": column,
        "unique_values":
            df[column].nunique(
                dropna=True
            ),

        "missing_percent":
            round(
                df[column]
                .isna()
                .mean()
                * 100,
                2
            ),

        "most_common_value":
            value_counts.index[0],

        "most_common_count":
            int(value_counts.iloc[0]),

        "most_common_percent":
            round(
                value_counts.iloc[0]
                / len(df)
                * 100,
                2
            )
    })


cardinality_summary = pd.DataFrame(
    cardinality_rows
)

cardinality_summary.to_csv(
    TABLE_DIR / "10_categorical_cardinality.csv",
    index=False
)


print("\n" + "=" * 85)
print("CATEGORICAL CARDINALITY")
print("=" * 85)

print(
    cardinality_summary
    .to_string(index=False)
)


# ============================================================
# 15. CATEGORY FRAUD RATE HELPER
# ============================================================

def category_fraud_table(
    dataframe,
    column,
    min_support
):

    temp = dataframe.copy()

    temp[column] = (
        temp[column]
        .fillna("Missing")
    )

    summary = (
        temp.groupby(column)
        ["fraudulent"]
        .agg(
            count="count",
            fraud_count="sum",
            fraud_rate="mean"
        )
    )

    summary["fraud_rate"] *= 100

    summary = (
        summary
        .sort_values(
            "fraud_rate",
            ascending=False
        )
    )

    stable = (
        summary[
            summary["count"] >= min_support
        ]
    )

    return summary, stable


# ============================================================
# 16. EMPLOYMENT TYPE
# ============================================================

employment_all, employment_stable = (
    category_fraud_table(
        df,
        "employment_type",
        min_support=30
    )
)

employment_all.to_csv(
    TABLE_DIR / "11_employment_type_fraud_rate.csv"
)

plt.figure(figsize=(9, 6))

plt.barh(
    employment_stable.index,
    employment_stable["fraud_rate"]
)

plt.title(
    "Fraud Rate by Employment Type"
)

plt.xlabel("Fraud rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_employment_type_fraud_rate.png",
    dpi=200
)

plt.close()


# ============================================================
# 17. EXPERIENCE
# ============================================================

experience_all, experience_stable = (
    category_fraud_table(
        df,
        "required_experience",
        min_support=30
    )
)

experience_all.to_csv(
    TABLE_DIR / "12_experience_fraud_rate.csv"
)

plt.figure(figsize=(10, 6))

plt.barh(
    experience_stable.index,
    experience_stable["fraud_rate"]
)

plt.title(
    "Fraud Rate by Required Experience"
)

plt.xlabel("Fraud rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "08_experience_fraud_rate.png",
    dpi=200
)

plt.close()


# ============================================================
# 18. EDUCATION
# ============================================================

education_all, education_stable = (
    category_fraud_table(
        df,
        "required_education",
        min_support=30
    )
)

education_all.to_csv(
    TABLE_DIR / "13_education_fraud_rate.csv"
)

plt.figure(figsize=(10, 7))

plt.barh(
    education_stable.index,
    education_stable["fraud_rate"]
)

plt.title(
    "Fraud Rate by Required Education"
)

plt.xlabel("Fraud rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_education_fraud_rate.png",
    dpi=200
)

plt.close()


# ============================================================
# 19. INDUSTRY
# ============================================================

industry_all, industry_stable = (
    category_fraud_table(
        df,
        "industry",
        min_support=50
    )
)

industry_all.to_csv(
    TABLE_DIR / "14_industry_fraud_rate.csv"
)


industry_top = (
    industry_stable
    .head(15)
    .sort_values(
        "fraud_rate"
    )
)


plt.figure(figsize=(11, 7))

plt.barh(
    industry_top.index,
    industry_top["fraud_rate"]
)

plt.title(
    "Industries with Highest Fraud Rate "
    "(Minimum 50 Postings)"
)

plt.xlabel("Fraud rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "10_industry_fraud_rate.png",
    dpi=200
)

plt.close()


# ============================================================
# 20. FUNCTION
# ============================================================

function_all, function_stable = (
    category_fraud_table(
        df,
        "function",
        min_support=50
    )
)

function_all.to_csv(
    TABLE_DIR / "15_function_fraud_rate.csv"
)


function_top = (
    function_stable
    .head(15)
    .sort_values(
        "fraud_rate"
    )
)


plt.figure(figsize=(10, 7))

plt.barh(
    function_top.index,
    function_top["fraud_rate"]
)

plt.title(
    "Job Functions with Highest Fraud Rate "
    "(Minimum 50 Postings)"
)

plt.xlabel("Fraud rate (%)")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "11_function_fraud_rate.png",
    dpi=200
)

plt.close()


# ============================================================
# 21. COUNTRY EXTRACTION
# ============================================================

df["country"] = (
    df["location"]
    .fillna("")
    .astype(str)
    .str.split(",")
    .str[0]
    .str.strip()
)


country_all, country_stable = (
    category_fraud_table(
        df[df["country"] != ""],
        "country",
        min_support=50
    )
)

country_all.to_csv(
    TABLE_DIR / "16_country_fraud_rate.csv"
)


# ============================================================
# 22. SALARY ANALYSIS
# ============================================================

df["has_salary"] = (
    df["salary_range"]
    .notna()
    &
    (
        df["salary_range"]
        .astype(str)
        .str.strip()
        != ""
    )
).astype(int)


def parse_salary_range(value):

    if pd.isna(value):
        return pd.Series(
            [np.nan, np.nan]
        )

    value = str(value)

    value = value.replace(",", "")

    numbers = re.findall(
        r"\d+(?:\.\d+)?",
        value
    )

    if len(numbers) >= 2:

        return pd.Series([
            float(numbers[0]),
            float(numbers[1])
        ])

    if len(numbers) == 1:

        number = float(numbers[0])

        return pd.Series([
            number,
            number
        ])

    return pd.Series(
        [np.nan, np.nan]
    )


df[
    [
        "salary_min_raw",
        "salary_max_raw"
    ]
] = (
    df["salary_range"]
    .apply(parse_salary_range)
)


df["salary_parseable"] = (
    df["salary_min_raw"]
    .notna()
    .astype(int)
)


salary_summary_rows = []

for label_value, label_name in [
    (0, "Real"),
    (1, "Fraudulent")
]:

    subset = df[
        df["fraudulent"] == label_value
    ]

    salary_summary_rows.append({
        "class": label_name,

        "records":
            len(subset),

        "salary_present_percent":
            round(
                subset["has_salary"]
                .mean()
                * 100,
                2
            ),

        "salary_parseable_percent":
            round(
                subset[
                    "salary_parseable"
                ]
                .mean()
                * 100,
                2
            ),

        "median_min_raw":
            subset[
                "salary_min_raw"
            ].median(),

        "median_max_raw":
            subset[
                "salary_max_raw"
            ].median()
    })


salary_summary = pd.DataFrame(
    salary_summary_rows
)

salary_summary.to_csv(
    TABLE_DIR / "17_salary_summary.csv",
    index=False
)


# Plot presence only.
# Raw amounts are not directly comparable across countries/currencies.

salary_presence = (
    df.groupby("has_salary")
    ["fraudulent"]
    .agg(
        count="count",
        fraud_rate="mean"
    )
)

salary_presence["fraud_rate"] *= 100

salary_presence.to_csv(
    TABLE_DIR / "18_salary_presence_fraud_rate.csv"
)


# ============================================================
# 23. CLEAN TEXT FOR EXPLORATORY TERM ANALYSIS
# ============================================================

def clean_for_terms(text):

    if pd.isna(text):
        return ""

    text = html.unescape(
        str(text)
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["_term_text"] = (
    df["title"]
    .fillna("")
    .astype(str)
    + " "
    +
    df["description"]
    .fillna("")
    .astype(str)
    + " "
    +
    df["requirements"]
    .fillna("")
    .astype(str)
)

df["_term_text"] = (
    df["_term_text"]
    .map(clean_for_terms)
)


# ============================================================
# 24. TOP TERM FUNCTION
# ============================================================

def get_top_terms(
    texts,
    ngram_range=(1, 1),
    top_n=20,
    min_df=3
):

    vectorizer = CountVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=30000
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    counts = np.asarray(
        matrix.sum(axis=0)
    ).ravel()

    terms = np.array(
        vectorizer
        .get_feature_names_out()
    )

    order = np.argsort(
        counts
    )[::-1][:top_n]

    return pd.DataFrame({
        "term": terms[order],
        "count": counts[order]
    })


# ============================================================
# 25. TOP WORDS - REAL
# ============================================================

real_terms = get_top_terms(
    df.loc[
        df["fraudulent"] == 0,
        "_term_text"
    ],
    ngram_range=(1, 1),
    top_n=20,
    min_df=5
)

real_terms.to_csv(
    TABLE_DIR / "19_top_unigrams_real.csv",
    index=False
)


plt.figure(figsize=(9, 7))

plt.barh(
    real_terms["term"][::-1],
    real_terms["count"][::-1]
)

plt.title(
    "Most Frequent Terms in Real Job Postings"
)

plt.xlabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "12_top_terms_real.png",
    dpi=200
)

plt.close()


# ============================================================
# 26. TOP WORDS - FRAUD
# ============================================================

fraud_terms = get_top_terms(
    df.loc[
        df["fraudulent"] == 1,
        "_term_text"
    ],
    ngram_range=(1, 1),
    top_n=20,
    min_df=3
)

fraud_terms.to_csv(
    TABLE_DIR / "20_top_unigrams_fraud.csv",
    index=False
)


plt.figure(figsize=(9, 7))

plt.barh(
    fraud_terms["term"][::-1],
    fraud_terms["count"][::-1]
)

plt.title(
    "Most Frequent Terms in Fraudulent Job Postings"
)

plt.xlabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "13_top_terms_fraud.png",
    dpi=200
)

plt.close()


# ============================================================
# 27. TOP BIGRAMS - FRAUD
# ============================================================

fraud_bigrams = get_top_terms(
    df.loc[
        df["fraudulent"] == 1,
        "_term_text"
    ],
    ngram_range=(2, 2),
    top_n=20,
    min_df=3
)

fraud_bigrams.to_csv(
    TABLE_DIR / "21_top_bigrams_fraud.csv",
    index=False
)


plt.figure(figsize=(10, 7))

plt.barh(
    fraud_bigrams["term"][::-1],
    fraud_bigrams["count"][::-1]
)

plt.title(
    "Most Frequent Bigrams in Fraudulent Job Postings"
)

plt.xlabel("Frequency")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "14_top_bigrams_fraud.png",
    dpi=200
)

plt.close()


# ============================================================
# 28. MANUAL SAMPLE INSPECTION
# ============================================================

manual_columns = [
    "job_id",
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
    "fraudulent"
]


real_sample = (
    df[
        df["fraudulent"] == 0
    ][manual_columns]
    .sample(
        n=20,
        random_state=42
    )
)


fraud_sample = (
    df[
        df["fraudulent"] == 1
    ][manual_columns]
    .sample(
        n=20,
        random_state=42
    )
)


real_sample.to_csv(
    SAMPLE_DIR / "real_jobs_sample_20.csv",
    index=False
)

fraud_sample.to_csv(
    SAMPLE_DIR / "fraud_jobs_sample_20.csv",
    index=False
)


# ============================================================
# 29. KEY FINDINGS OUTPUT
# ============================================================

print("\n" + "=" * 85)
print("KEY EDA CHECKS")
print("=" * 85)

print("\nTarget distribution:")
print(target_summary)


print("\nTop missing-value gaps between Fraud and Real:")

print(
    missing_by_class[
        [
            "real_missing_percent",
            "fraud_missing_percent",
            "gap_fraud_minus_real"
        ]
    ]
    .drop(
        index="fraudulent",
        errors="ignore"
    )
    .head(10)
)


print("\nBinary feature fraud rates:")

print(
    binary_summary[
        [
            "feature",
            "count",
            "fraud_count",
            "fraud_rate"
        ]
    ]
    .to_string(
        index=False
    )
)


print("\nSalary summary:")

print(
    salary_summary.to_string(
        index=False
    )
)


print("\nTop fraudulent terms:")

print(
    fraud_terms.to_string(
        index=False
    )
)


print("\nTop fraudulent bigrams:")

print(
    fraud_bigrams.to_string(
        index=False
    )
)


# ============================================================
# 30. FINISH
# ============================================================

print("\n" + "=" * 85)
print("COMPLETE EDA FINISHED SUCCESSFULLY")
print("=" * 85)

print(
    f"\nFigures:\n{FIGURE_DIR}"
)

print(
    f"\nTables:\n{TABLE_DIR}"
)

print(
    f"\nManual samples:\n{SAMPLE_DIR}"
)