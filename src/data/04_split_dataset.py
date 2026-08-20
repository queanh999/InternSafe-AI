from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# INTERNSAFE AI
# STEP 04 - LEAKAGE-SAFE TRAIN / VALIDATION / TEST SPLIT
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "emscad_preprocessed_base.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

MANIFEST_PATH = (
    OUTPUT_DIR
    / "split_manifest.csv"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "split_summary.csv"
)


# ============================================================
# 1. LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("=" * 85)
print("INTERNSAFE AI - LEAKAGE-SAFE DATA SPLIT")
print("=" * 85)

print(f"\nInput rows: {len(df):,}")

overall_fraud_rate = (
    df["fraudulent"].mean()
)

print(
    f"Overall fraud rate: "
    f"{overall_fraud_rate * 100:.2f}%"
)


# ============================================================
# 2. VALIDATE CONTENT GROUPS
# ============================================================

group_label_check = (
    df.groupby("content_group")
    ["fraudulent"]
    .nunique()
)

conflicting_groups = (
    group_label_check > 1
).sum()


if conflicting_groups != 0:
    raise ValueError(
        f"Found {conflicting_groups} groups "
        "with conflicting labels."
    )


# ============================================================
# 3. CREATE ONE ROW PER CONTENT GROUP
# ============================================================

groups = (
    df.groupby("content_group")
    .agg(
        fraudulent=(
            "fraudulent",
            "first"
        ),
        group_size=(
            "job_id",
            "size"
        )
    )
    .reset_index()
)


print(
    f"Unique content groups: "
    f"{len(groups):,}"
)


print("\nGroup label distribution:")

print(
    groups["fraudulent"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 4. SPLIT FUNCTION
# ============================================================

def create_split(seed):

    # --------------------------------------------------------
    # TRAIN = 70%
    # TEMP = 30%
    # --------------------------------------------------------

    train_groups, temp_groups = (
        train_test_split(
            groups,
            test_size=0.30,
            random_state=seed,
            stratify=groups[
                "fraudulent"
            ]
        )
    )


    # --------------------------------------------------------
    # VALIDATION = 15%
    # TEST = 15%
    # --------------------------------------------------------

    validation_groups, test_groups = (
        train_test_split(
            temp_groups,
            test_size=0.50,
            random_state=seed,
            stratify=temp_groups[
                "fraudulent"
            ]
        )
    )


    train_ids = set(
        train_groups[
            "content_group"
        ]
    )

    validation_ids = set(
        validation_groups[
            "content_group"
        ]
    )

    test_ids = set(
        test_groups[
            "content_group"
        ]
    )


    train_df = df[
        df["content_group"]
        .isin(train_ids)
    ].copy()


    validation_df = df[
        df["content_group"]
        .isin(validation_ids)
    ].copy()


    test_df = df[
        df["content_group"]
        .isin(test_ids)
    ].copy()


    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# 5. SEARCH FOR A GOOD RANDOM SEED
# ============================================================

target_ratios = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}


best_seed = None
best_score = float("inf")
best_result = None


for seed in range(200):

    (
        train_candidate,
        validation_candidate,
        test_candidate
    ) = create_split(seed)


    candidates = {
        "train": train_candidate,
        "validation":
            validation_candidate,
        "test": test_candidate
    }


    score = 0.0


    for name, part in (
        candidates.items()
    ):

        # --------------------------------------------
        # Difference from desired dataset size
        # --------------------------------------------

        actual_ratio = (
            len(part) / len(df)
        )

        ratio_error = abs(
            actual_ratio
            - target_ratios[name]
        )


        # --------------------------------------------
        # Difference from global fraud rate
        # --------------------------------------------

        fraud_error = abs(
            part["fraudulent"].mean()
            - overall_fraud_rate
        )


        # Fraud-rate preservation matters more
        score += (
            ratio_error
            + 2.0 * fraud_error
        )


    if score < best_score:

        best_score = score
        best_seed = seed

        best_result = (
            train_candidate,
            validation_candidate,
            test_candidate
        )


print(
    f"\nBest random seed: "
    f"{best_seed}"
)


train_df, validation_df, test_df = (
    best_result
)


# ============================================================
# 6. GROUP-LEAKAGE CHECK
# ============================================================

train_groups = set(
    train_df["content_group"]
)

validation_groups = set(
    validation_df[
        "content_group"
    ]
)

test_groups = set(
    test_df["content_group"]
)


train_validation_overlap = (
    train_groups
    & validation_groups
)

train_test_overlap = (
    train_groups
    & test_groups
)

validation_test_overlap = (
    validation_groups
    & test_groups
)


print("\n" + "=" * 85)
print("GROUP LEAKAGE CHECK")
print("=" * 85)


print(
    "Train ∩ Validation:",
    len(train_validation_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(validation_test_overlap)
)


if (
    train_validation_overlap
    or train_test_overlap
    or validation_test_overlap
):

    raise ValueError(
        "DATA LEAKAGE DETECTED."
    )


# ============================================================
# 7. JOB ID OVERLAP CHECK
# ============================================================

train_job_ids = set(
    train_df["job_id"]
)

validation_job_ids = set(
    validation_df["job_id"]
)

test_job_ids = set(
    test_df["job_id"]
)


if (
    train_job_ids & validation_job_ids
    or train_job_ids & test_job_ids
    or validation_job_ids & test_job_ids
):

    raise ValueError(
        "job_id overlap detected."
    )


# ============================================================
# 8. VERIFY ALL ROWS ARE PRESENT
# ============================================================

total_rows_after_split = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)


if total_rows_after_split != len(df):

    raise ValueError(
        "Some rows were lost or duplicated "
        "during splitting."
    )


# ============================================================
# 9. SPLIT SUMMARY
# ============================================================

def summarize_split(
    name,
    dataframe
):

    return {
        "split": name,

        "rows":
            len(dataframe),

        "row_percent":
            round(
                len(dataframe)
                / len(df)
                * 100,
                2
            ),

        "unique_groups":
            dataframe[
                "content_group"
            ].nunique(),

        "real_count":
            int(
                (
                    dataframe[
                        "fraudulent"
                    ] == 0
                ).sum()
            ),

        "fraud_count":
            int(
                (
                    dataframe[
                        "fraudulent"
                    ] == 1
                ).sum()
            ),

        "fraud_rate_percent":
            round(
                dataframe[
                    "fraudulent"
                ].mean()
                * 100,
                2
            )
    }


summary = pd.DataFrame([
    summarize_split(
        "Train",
        train_df
    ),

    summarize_split(
        "Validation",
        validation_df
    ),

    summarize_split(
        "Test",
        test_df
    )
])


print("\n" + "=" * 85)
print("SPLIT SUMMARY")
print("=" * 85)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE DATASETS
# ============================================================

train_df.to_csv(
    TRAIN_PATH,
    index=False
)

validation_df.to_csv(
    VALIDATION_PATH,
    index=False
)

test_df.to_csv(
    TEST_PATH,
    index=False
)


summary.to_csv(
    SUMMARY_PATH,
    index=False
)


# ============================================================
# 11. SAVE SPLIT MANIFEST
# ============================================================

train_manifest = train_df[
    [
        "job_id",
        "content_group",
        "fraudulent"
    ]
].copy()

train_manifest[
    "split"
] = "train"


validation_manifest = (
    validation_df[
        [
            "job_id",
            "content_group",
            "fraudulent"
        ]
    ].copy()
)

validation_manifest[
    "split"
] = "validation"


test_manifest = test_df[
    [
        "job_id",
        "content_group",
        "fraudulent"
    ]
].copy()

test_manifest[
    "split"
] = "test"


manifest = pd.concat(
    [
        train_manifest,
        validation_manifest,
        test_manifest
    ],
    ignore_index=True
)


manifest.to_csv(
    MANIFEST_PATH,
    index=False
)


# ============================================================
# 12. FINAL CHECK
# ============================================================

print("\n" + "=" * 85)
print("DATA SPLIT COMPLETED SUCCESSFULLY")
print("=" * 85)

print(
    f"\nTrain:\n{TRAIN_PATH}"
)

print(
    f"\nValidation:\n"
    f"{VALIDATION_PATH}"
)

print(
    f"\nTest:\n{TEST_PATH}"
)

print(
    f"\nManifest:\n"
    f"{MANIFEST_PATH}"
)