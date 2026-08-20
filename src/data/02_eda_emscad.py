from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INTERNSAFE AI
# STEP 02 - FULL EXPLORATORY DATA ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fake_job_postings.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("INTERNSAFE AI - FULL EMSCAD EDA")
print("=" * 80)

print(f"\nDataset shape: {df.shape}")
print(f"Fraud rate: {df['fraudulent'].mean() * 100:.2f}%")


# ============================================================
# 3. CREATE ANALYSIS FEATURES
# ============================================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

for col in text_columns:

    text = (
        df[col]
        .fillna("")
        .astype(str)
    )

    df[f"{col}_char_length"] = text.str.len()

    df[f"{col}_word_length"] = (
        text
        .str.split()
        .str.len()
    )


# Missing-value indicators
df["missing_company_profile"] = (
    df["company_profile"].isna().astype(int)
)

df["missing_requirements"] = (
    df["requirements"].isna().astype(int)
)

df["missing_benefits"] = (
    df["benefits"].isna().astype(int)
)

df["missing_salary"] = (
    df["salary_range"].isna().astype(int)
)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

target_counts = (
    df["fraudulent"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(7, 5))

target_counts.plot(kind="bar")

plt.title("Distribution of Real and Fraudulent Job Postings")
plt.xlabel("Class")
plt.ylabel("Number of job postings")

plt.xticks(
    ticks=[0, 1],
    labels=["Real (0)", "Fraudulent (1)"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "01_target_distribution.png",
    dpi=200
)

plt.close()

print("Saved: 01_target_distribution.png")


# Save target summary
target_summary = pd.DataFrame({
    "count": target_counts,
    "percent": (
        target_counts
        / len(df)
        * 100
    ).round(2)
})

target_summary.to_csv(
    TABLE_DIR / "01_target_summary.csv"
)


# ============================================================
# 5. MISSING VALUES
# ============================================================

missing_summary = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (
        df.isna().mean() * 100
    ).round(2)
})

missing_summary = (
    missing_summary
    .sort_values(
        "missing_percent",
        ascending=False
    )
)

missing_summary.to_csv(
    TABLE_DIR / "02_missing_summary.csv"
)

missing_plot = missing_summary[
    missing_summary["missing_percent"] > 0
]

plt.figure(figsize=(11, 6))

missing_plot[
    "missing_percent"
].plot(kind="bar")

plt.title("Missing Values in EMSCAD Features")
plt.xlabel("Feature")
plt.ylabel("Missing values (%)")

plt.xticks(
    rotation=60,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "02_missing_values.png",
    dpi=200
)

plt.close()

print("Saved: 02_missing_values.png")


# ============================================================
# 6. EXACT CONTENT DUPLICATES
# ============================================================

print("\n" + "=" * 80)
print("CONTENT DUPLICATE CHECK")
print("=" * 80)

description_duplicates = (
    df["description"]
    .fillna("")
    .duplicated()
    .sum()
)

combined_for_duplicate = (
    df["title"].fillna("").astype(str)
    + " "
    + df["description"].fillna("").astype(str)
)

combined_duplicates = (
    combined_for_duplicate
    .duplicated()
    .sum()
)

print(
    f"Duplicate descriptions: "
    f"{description_duplicates:,}"
)

print(
    f"Duplicate title + description: "
    f"{combined_duplicates:,}"
)

duplicate_summary = pd.DataFrame({
    "type": [
        "description",
        "title_plus_description"
    ],
    "duplicate_count": [
        description_duplicates,
        combined_duplicates
    ]
})

duplicate_summary.to_csv(
    TABLE_DIR / "03_content_duplicates.csv",
    index=False
)


# ============================================================
# 7. FRAUD RATE BY BINARY FEATURES
# ============================================================

binary_features = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "missing_company_profile",
    "missing_requirements",
    "missing_benefits",
    "missing_salary",
]

binary_results = []

for feature in binary_features:

    result = (
        df.groupby(feature)
        ["fraudulent"]
        .agg(
            count="count",
            fraud_rate="mean"
        )
        .reset_index()
    )

    result["fraud_rate"] *= 100
    result["feature"] = feature

    binary_results.append(result)


binary_summary = pd.concat(
    binary_results,
    ignore_index=True
)

binary_summary.to_csv(
    TABLE_DIR / "04_binary_feature_fraud_rates.csv",
    index=False
)


# Plot selected binary features
plot_features = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "missing_company_profile",
]

fig_data = []

for feature in plot_features:

    rates = (
        df.groupby(feature)
        ["fraudulent"]
        .mean()
        .mul(100)
    )

    for value, rate in rates.items():

        fig_data.append({
            "feature": feature,
            "value": value,
            "fraud_rate": rate
        })


fig_df = pd.DataFrame(fig_data)

labels = (
    fig_df["feature"]
    + "="
    + fig_df["value"].astype(str)
)

plt.figure(figsize=(11, 6))

plt.bar(
    labels,
    fig_df["fraud_rate"]
)

plt.title("Fraud Rate by Binary Job Features")
plt.xlabel("Feature value")
plt.ylabel("Fraud rate (%)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "03_binary_feature_fraud_rate.png",
    dpi=200
)

plt.close()

print(
    "Saved: 03_binary_feature_fraud_rate.png"
)


# ============================================================
# 8. TEXT LENGTH SUMMARY
# ============================================================

length_features = [
    "title_word_length",
    "company_profile_word_length",
    "description_word_length",
    "requirements_word_length",
    "benefits_word_length",
]

text_length_summary = (
    df.groupby("fraudulent")[
        length_features
    ]
    .mean()
    .round(2)
)

text_length_summary.to_csv(
    TABLE_DIR / "05_text_length_summary.csv"
)

print("\n" + "=" * 80)
print("AVERAGE TEXT LENGTH BY CLASS")
print("=" * 80)

print(text_length_summary)


# ============================================================
# 9. DESCRIPTION LENGTH BY TARGET
# ============================================================

real_description = (
    df.loc[
        df["fraudulent"] == 0,
        "description_word_length"
    ]
)

fraud_description = (
    df.loc[
        df["fraudulent"] == 1,
        "description_word_length"
    ]
)


# Limit extreme values for visualization only
plot_limit = (
    df["description_word_length"]
    .quantile(0.99)
)

real_plot = real_description[
    real_description <= plot_limit
]

fraud_plot = fraud_description[
    fraud_description <= plot_limit
]


plt.figure(figsize=(9, 6))

plt.hist(
    real_plot,
    bins=50,
    alpha=0.6,
    label="Real"
)

plt.hist(
    fraud_plot,
    bins=50,
    alpha=0.6,
    label="Fraudulent"
)

plt.title(
    "Description Length: Real vs Fraudulent Jobs"
)

plt.xlabel(
    "Description length (words)"
)

plt.ylabel(
    "Frequency"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "04_description_length_by_target.png",
    dpi=200
)

plt.close()

print(
    "Saved: 04_description_length_by_target.png"
)


# ============================================================
# 10. EMPLOYMENT TYPE FRAUD RATE
# ============================================================

employment_summary = (
    df.assign(
        employment_type=df[
            "employment_type"
        ].fillna("Missing")
    )
    .groupby("employment_type")
    ["fraudulent"]
    .agg(
        count="count",
        fraud_rate="mean"
    )
)

employment_summary["fraud_rate"] *= 100

employment_summary = (
    employment_summary
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

employment_summary.to_csv(
    TABLE_DIR / "06_employment_type_fraud_rate.csv"
)

plt.figure(figsize=(9, 6))

employment_summary[
    "fraud_rate"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Fraud Rate by Employment Type"
)

plt.xlabel(
    "Fraud rate (%)"
)

plt.ylabel(
    "Employment type"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_employment_type_fraud_rate.png",
    dpi=200
)

plt.close()

print(
    "Saved: 05_employment_type_fraud_rate.png"
)


# ============================================================
# 11. REQUIRED EXPERIENCE FRAUD RATE
# ============================================================

experience_summary = (
    df.assign(
        required_experience=df[
            "required_experience"
        ].fillna("Missing")
    )
    .groupby("required_experience")
    ["fraudulent"]
    .agg(
        count="count",
        fraud_rate="mean"
    )
)

experience_summary["fraud_rate"] *= 100

experience_summary = (
    experience_summary
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

experience_summary.to_csv(
    TABLE_DIR / "07_experience_fraud_rate.csv"
)

plt.figure(figsize=(10, 6))

experience_summary[
    "fraud_rate"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Fraud Rate by Required Experience"
)

plt.xlabel(
    "Fraud rate (%)"
)

plt.ylabel(
    "Required experience"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "06_experience_fraud_rate.png",
    dpi=200
)

plt.close()

print(
    "Saved: 06_experience_fraud_rate.png"
)


# ============================================================
# 12. REQUIRED EDUCATION FRAUD RATE
# ============================================================

education_summary = (
    df.assign(
        required_education=df[
            "required_education"
        ].fillna("Missing")
    )
    .groupby("required_education")
    ["fraudulent"]
    .agg(
        count="count",
        fraud_rate="mean"
    )
)

education_summary["fraud_rate"] *= 100

# Avoid tiny groups for visualization
education_plot = (
    education_summary[
        education_summary["count"] >= 30
    ]
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

education_summary.to_csv(
    TABLE_DIR / "08_education_fraud_rate.csv"
)

plt.figure(figsize=(11, 7))

education_plot[
    "fraud_rate"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Fraud Rate by Required Education"
)

plt.xlabel(
    "Fraud rate (%)"
)

plt.ylabel(
    "Required education"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "07_education_fraud_rate.png",
    dpi=200
)

plt.close()

print(
    "Saved: 07_education_fraud_rate.png"
)


# ============================================================
# 13. INDUSTRY FRAUD RATE
# ============================================================

industry_summary = (
    df.assign(
        industry=df[
            "industry"
        ].fillna("Missing")
    )
    .groupby("industry")
    ["fraudulent"]
    .agg(
        count="count",
        fraud_count="sum",
        fraud_rate="mean"
    )
)

industry_summary["fraud_rate"] *= 100

industry_summary = (
    industry_summary
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

industry_summary.to_csv(
    TABLE_DIR / "09_industry_fraud_rate.csv"
)


# Only categories with enough samples
industry_plot = (
    industry_summary[
        industry_summary["count"] >= 50
    ]
    .sort_values(
        "fraud_rate",
        ascending=False
    )
    .head(15)
)

plt.figure(figsize=(11, 7))

industry_plot[
    "fraud_rate"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Industries with Highest Fraud Rate "
    "(minimum 50 postings)"
)

plt.xlabel(
    "Fraud rate (%)"
)

plt.ylabel(
    "Industry"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "08_industry_fraud_rate.png",
    dpi=200
)

plt.close()

print(
    "Saved: 08_industry_fraud_rate.png"
)


# ============================================================
# 14. FUNCTION FRAUD RATE
# ============================================================

function_summary = (
    df.assign(
        function=df[
            "function"
        ].fillna("Missing")
    )
    .groupby("function")
    ["fraudulent"]
    .agg(
        count="count",
        fraud_count="sum",
        fraud_rate="mean"
    )
)

function_summary["fraud_rate"] *= 100

function_summary = (
    function_summary
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

function_summary.to_csv(
    TABLE_DIR / "10_function_fraud_rate.csv"
)


function_plot = (
    function_summary[
        function_summary["count"] >= 50
    ]
    .sort_values(
        "fraud_rate",
        ascending=False
    )
    .head(15)
)

plt.figure(figsize=(10, 7))

function_plot[
    "fraud_rate"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Job Functions with Highest Fraud Rate "
    "(minimum 50 postings)"
)

plt.xlabel(
    "Fraud rate (%)"
)

plt.ylabel(
    "Job function"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "09_function_fraud_rate.png",
    dpi=200
)

plt.close()

print(
    "Saved: 09_function_fraud_rate.png"
)


# ============================================================
# 15. FRAUD VS COMPANY LOGO
# ============================================================

logo_crosstab = pd.crosstab(
    df["has_company_logo"],
    df["fraudulent"],
    normalize="index"
) * 100

logo_crosstab.to_csv(
    TABLE_DIR / "11_logo_target_crosstab.csv"
)

logo_crosstab.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title(
    "Real/Fraud Distribution by Company Logo"
)

plt.xlabel(
    "Has company logo"
)

plt.ylabel(
    "Percentage within group"
)

plt.xticks(
    rotation=0
)

plt.legend(
    ["Real", "Fraudulent"]
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "10_company_logo_vs_target.png",
    dpi=200
)

plt.close()

print(
    "Saved: 10_company_logo_vs_target.png"
)


# ============================================================
# 16. COUNTRY ANALYSIS
# ============================================================

df["country"] = (
    df["location"]
    .fillna("")
    .astype(str)
    .str.split(",")
    .str[0]
    .str.strip()
)

country_summary = (
    df[df["country"] != ""]
    .groupby("country")
    ["fraudulent"]
    .agg(
        count="count",
        fraud_count="sum",
        fraud_rate="mean"
    )
)

country_summary["fraud_rate"] *= 100

country_summary = (
    country_summary
    .sort_values(
        "count",
        ascending=False
    )
)

country_summary.to_csv(
    TABLE_DIR / "12_country_summary.csv"
)


top_countries = (
    country_summary
    .head(15)
)

plt.figure(figsize=(10, 6))

top_countries[
    "count"
].sort_values().plot(
    kind="barh"
)

plt.title(
    "Top Countries by Number of Job Postings"
)

plt.xlabel(
    "Number of postings"
)

plt.ylabel(
    "Country"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "11_country_distribution.png",
    dpi=200
)

plt.close()

print(
    "Saved: 11_country_distribution.png"
)


# ============================================================
# 17. HIGH-LEVEL DATASET SUMMARY
# ============================================================

overview = pd.DataFrame({
    "metric": [
        "total_records",
        "real_records",
        "fraud_records",
        "fraud_rate_percent",
        "duplicate_rows",
        "duplicate_descriptions",
        "duplicate_title_description"
    ],
    "value": [
        len(df),
        int((df["fraudulent"] == 0).sum()),
        int((df["fraudulent"] == 1).sum()),
        round(df["fraudulent"].mean() * 100, 2),
        int(df.duplicated().sum()),
        int(description_duplicates),
        int(combined_duplicates)
    ]
})

overview.to_csv(
    TABLE_DIR / "00_dataset_overview.csv",
    index=False
)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETED")
print("=" * 80)

print(
    f"\nFigures saved to:\n{FIGURE_DIR}"
)

print(
    f"\nTables saved to:\n{TABLE_DIR}"
)