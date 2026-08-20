from pathlib import Path
import time
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# INTERNSAFE AI
# STEP 08 - HYPERPARAMETER TUNING
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

RESULT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "tuning"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "tuned"
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
# 2. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VALIDATION_PATH)

print("=" * 95)
print("INTERNSAFE AI - HYPERPARAMETER TUNING")
print("=" * 95)

print(f"\nTrain      : {train_df.shape}")
print(f"Validation : {val_df.shape}")


# ============================================================
# 3. TARGET + GROUPS
# ============================================================

y_train = (
    train_df["fraudulent"]
    .astype(int)
)

y_val = (
    val_df["fraudulent"]
    .astype(int)
)

groups = train_df["content_group"]


# ============================================================
# 4. FEATURES
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


# ============================================================
# 5. BUILD PREPROCESSOR
# ============================================================

def build_preprocessor():

    text_transformer = TfidfVectorizer(

        lowercase=True,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.98,

        max_features=50000,

        sublinear_tf=True,

        token_pattern=r"(?u)\b\w\w+\b",
    )


    numeric_transformer = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",

            StandardScaler(
                with_mean=False
            )
        ),
    ])


    categorical_transformer = Pipeline([

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",

            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=10
            )
        ),
    ])


    return ColumnTransformer([

        (
            "text",
            text_transformer,
            text_feature
        ),

        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),

        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
    ])


# ============================================================
# 6. GROUP-AWARE CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(

    n_splits=3,

    shuffle=True,

    random_state=42
)


scoring = {

    "pr_auc":
        "average_precision",

    "roc_auc":
        "roc_auc",

    "fraud_f1":
        "f1",

    "fraud_recall":
        "recall",

    "fraud_precision":
        "precision",
}


# ============================================================
# 7. VALIDATION EVALUATION
# ============================================================

def evaluate_validation(
    name,
    model
):

    predictions = model.predict(
        X_val
    )


    if hasattr(
        model,
        "predict_proba"
    ):

        scores = (
            model
            .predict_proba(
                X_val
            )[:, 1]
        )

    else:

        scores = (
            model
            .decision_function(
                X_val
            )
        )


    cm = confusion_matrix(
        y_val,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()


    result = {

        "model":
            name,

        "accuracy":
            accuracy_score(
                y_val,
                predictions
            ),

        "fraud_precision":
            precision_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "fraud_recall":
            recall_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "fraud_f1":
            f1_score(
                y_val,
                predictions,
                zero_division=0
            ),

        "macro_f1":
            f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0
            ),

        "roc_auc":
            roc_auc_score(
                y_val,
                scores
            ),

        "pr_auc":
            average_precision_score(
                y_val,
                scores
            ),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


    return result


# ============================================================
# 8. SVM SEARCH
# ============================================================

print("\n" + "=" * 95)
print("TUNING COMBINED LINEAR SVM")
print("=" * 95)


svm_pipeline = Pipeline([

    (
        "preprocessor",
        build_preprocessor()
    ),

    (
        "classifier",

        LinearSVC(

            class_weight="balanced",

            max_iter=30000,

            random_state=42
        )
    ),
])


svm_parameters = {

    "preprocessor__text__ngram_range": [
        (1, 1),
        (1, 2),
        (1, 3),
    ],

    "preprocessor__text__min_df": [
        1,
        2,
        3,
        5,
    ],

    "preprocessor__text__max_features": [
        30000,
        50000,
        70000,
    ],

    "preprocessor__text__sublinear_tf": [
        True,
        False,
    ],

    "classifier__C": [
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
    ],

    "classifier__class_weight": [

        "balanced",

        {
            0: 1.0,
            1: 8.0
        },

        {
            0: 1.0,
            1: 12.0
        },

        {
            0: 1.0,
            1: 16.0
        },
    ],
}


svm_search = RandomizedSearchCV(

    estimator=svm_pipeline,

    param_distributions=
        svm_parameters,

    n_iter=10,

    scoring=scoring,

    # Primary tuning objective
    refit="pr_auc",

    cv=cv,

    random_state=42,

    n_jobs=-1,

    verbose=2,

    return_train_score=False,
)


svm_start = time.time()


svm_search.fit(

    X_train,

    y_train,

    groups=groups
)


svm_seconds = (
    time.time()
    - svm_start
)


print(
    f"\nSVM tuning time: "
    f"{svm_seconds:.2f} seconds"
)


print("\nBest SVM parameters:")

for key, value in (
    svm_search
    .best_params_
    .items()
):

    print(
        f"{key}: {value}"
    )


print(
    f"\nBest CV PR-AUC: "
    f"{svm_search.best_score_:.4f}"
)


# Save full CV table
svm_cv = pd.DataFrame(
    svm_search.cv_results_
)

svm_cv.to_csv(
    RESULT_DIR
    / "svm_random_search_results.csv",
    index=False
)


best_svm = (
    svm_search
    .best_estimator_
)


joblib.dump(
    best_svm,

    MODEL_DIR
    / "best_combined_svm.joblib"
)


# ============================================================
# 9. LOGISTIC SEARCH
# ============================================================

print("\n" + "=" * 95)
print("TUNING COMBINED LOGISTIC REGRESSION")
print("=" * 95)


logistic_pipeline = Pipeline([

    (
        "preprocessor",
        build_preprocessor()
    ),

    (
        "classifier",

        LogisticRegression(

            max_iter=5000,

            solver="liblinear",

            random_state=42
        )
    ),
])


logistic_parameters = {

    "preprocessor__text__ngram_range": [
        (1, 1),
        (1, 2),
        (1, 3),
    ],

    "preprocessor__text__min_df": [
        1,
        2,
        3,
        5,
    ],

    "preprocessor__text__max_features": [
        30000,
        50000,
        70000,
    ],

    "preprocessor__text__sublinear_tf": [
        True,
        False,
    ],

    "classifier__C": [
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
    ],

    "classifier__class_weight": [

        None,

        "balanced",

        {
            0: 1.0,
            1: 8.0
        },

        {
            0: 1.0,
            1: 12.0
        },

        {
            0: 1.0,
            1: 16.0
        },
    ],
}


logistic_search = RandomizedSearchCV(

    estimator=logistic_pipeline,

    param_distributions=
        logistic_parameters,

    n_iter=10,

    scoring=scoring,

    refit="pr_auc",

    cv=cv,

    random_state=43,

    n_jobs=-1,

    verbose=2,

    return_train_score=False,
)


logistic_start = time.time()


logistic_search.fit(

    X_train,

    y_train,

    groups=groups
)


logistic_seconds = (
    time.time()
    - logistic_start
)


print(
    f"\nLogistic tuning time: "
    f"{logistic_seconds:.2f} seconds"
)


print("\nBest Logistic parameters:")

for key, value in (
    logistic_search
    .best_params_
    .items()
):

    print(
        f"{key}: {value}"
    )


print(
    f"\nBest CV PR-AUC: "
    f"{logistic_search.best_score_:.4f}"
)


logistic_cv = pd.DataFrame(
    logistic_search.cv_results_
)

logistic_cv.to_csv(

    RESULT_DIR
    / "logistic_random_search_results.csv",

    index=False
)


best_logistic = (
    logistic_search
    .best_estimator_
)


joblib.dump(

    best_logistic,

    MODEL_DIR
    / "best_combined_logistic.joblib"
)


# ============================================================
# 10. EVALUATE BEST MODELS ON VALIDATION
# ============================================================

print("\n" + "=" * 95)
print("VALIDATION RESULTS AFTER TUNING")
print("=" * 95)


svm_validation = (
    evaluate_validation(
        "Tuned Combined SVM",
        best_svm
    )
)


logistic_validation = (
    evaluate_validation(
        "Tuned Combined Logistic",
        best_logistic
    )
)


validation_results = pd.DataFrame([

    svm_validation,

    logistic_validation
])


validation_results = (
    validation_results
    .sort_values(
        "pr_auc",
        ascending=False
    )
)


validation_results.to_csv(

    RESULT_DIR
    / "tuned_validation_results.csv",

    index=False
)


display_columns = [

    "model",

    "accuracy",

    "fraud_precision",

    "fraud_recall",

    "fraud_f1",

    "macro_f1",

    "roc_auc",

    "pr_auc",

    "fp",

    "fn",
]


print(

    validation_results[
        display_columns
    ]
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE PARAMETERS
# ============================================================

params = {

    "svm": {

        key: str(value)

        for key, value
        in svm_search
        .best_params_
        .items()
    },

    "logistic": {

        key: str(value)

        for key, value
        in logistic_search
        .best_params_
        .items()
    }
}


with open(

    RESULT_DIR
    / "best_parameters.json",

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        params,

        file,

        indent=4,

        ensure_ascii=False
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 95)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 95)

print(
    f"\nResults:\n"
    f"{RESULT_DIR}"
)

print(
    f"\nTuned models:\n"
    f"{MODEL_DIR}"
)