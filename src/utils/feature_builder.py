import html
import re

import pandas as pd


# ============================================================
# INTERNSAFE AI
# RAW JOB -> MODEL FEATURES
# ============================================================


def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = html.unescape(text)

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URLTOKEN ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\."
        r"[A-Za-z]{2,}\b",
        " EMAILTOKEN ",
        text
    )

    text = re.sub(
        r"\+?\d[\d\s().-]{7,}\d",
        " PHONETOKEN ",
        text
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def count_all_caps_words(text):

    if text is None:
        return 0

    return len(
        re.findall(
            r"\b[A-Z]{3,}\b",
            str(text)
        )
    )


def is_missing(value):

    if value is None:
        return True

    value = str(value).strip()

    return value == ""


def categorical_value(value):

    if is_missing(value):
        return "Missing"

    return str(value).strip()


def binary_value(value):

    if value in [1, True, "1", "true", "True", "yes", "Yes"]:
        return 1

    return 0


def build_features(job):

    # --------------------------------------------------------
    # RAW FIELDS
    # --------------------------------------------------------

    title = job.get("title", "")
    company_profile = job.get("company_profile", "")
    description = job.get("description", "")
    requirements = job.get("requirements", "")
    benefits = job.get("benefits", "")

    salary_range = job.get("salary_range", "")
    department = job.get("department", "")
    employment_type = job.get("employment_type", "")
    required_experience = job.get("required_experience", "")
    required_education = job.get("required_education", "")
    industry = job.get("industry", "")
    function = job.get("function", "")
    location = job.get("location", "")


    # --------------------------------------------------------
    # RAW COMBINED TEXT
    # --------------------------------------------------------

    raw_text = " ".join([
        str(title or ""),
        str(company_profile or ""),
        str(description or ""),
        str(requirements or ""),
        str(benefits or "")
    ])


    # --------------------------------------------------------
    # RAW TEXT SIGNALS
    # --------------------------------------------------------

    has_url = int(
        bool(
            re.search(
                r"https?://|www\.",
                raw_text,
                flags=re.IGNORECASE
            )
        )
    )

    has_email = int(
        bool(
            re.search(
                r"\b[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+\."
                r"[A-Za-z]{2,}\b",
                raw_text
            )
        )
    )

    has_phone_like = int(
        bool(
            re.search(
                r"\+?\d[\d\s().-]{7,}\d",
                raw_text
            )
        )
    )

    has_html = int(
        bool(
            re.search(
                r"<[^>]+>",
                raw_text
            )
        )
    )

    has_currency_symbol = int(
        bool(
            re.search(
                r"[$€£]",
                raw_text
            )
        )
    )


    # --------------------------------------------------------
    # CLEAN TEXT
    # --------------------------------------------------------

    title_clean = clean_text(title)
    company_profile_clean = clean_text(company_profile)
    description_clean = clean_text(description)
    requirements_clean = clean_text(requirements)
    benefits_clean = clean_text(benefits)


    # --------------------------------------------------------
    # COMBINED MODEL TEXT
    # --------------------------------------------------------

    combined_text = (

        "TITLE "
        + title_clean

        + " COMPANY "
        + company_profile_clean

        + " DESCRIPTION "
        + description_clean

        + " REQUIREMENTS "
        + requirements_clean

        + " BENEFITS "
        + benefits_clean
    )


    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    location_text = str(location or "").strip()

    if location_text:

        country = (
            location_text
            .split(",")[0]
            .strip()
        )

        if not country:
            country = "Missing"

    else:

        country = "Missing"


    # --------------------------------------------------------
    # LENGTH HELPERS
    # --------------------------------------------------------

    def chars(text):
        return len(text)

    def words(text):
        return len(text.split())


    # --------------------------------------------------------
    # CREATE ROW
    # --------------------------------------------------------

    row = {

        "combined_text":
            combined_text,

        # Original binary features
        "telecommuting":
            binary_value(
                job.get(
                    "telecommuting",
                    0
                )
            ),

        "has_company_logo":
            binary_value(
                job.get(
                    "has_company_logo",
                    0
                )
            ),

        "has_questions":
            binary_value(
                job.get(
                    "has_questions",
                    0
                )
            ),


        # Missing indicators
        "missing_company_profile":
            int(
                is_missing(
                    company_profile
                )
            ),

        "missing_requirements":
            int(
                is_missing(
                    requirements
                )
            ),

        "missing_benefits":
            int(
                is_missing(
                    benefits
                )
            ),

        "missing_salary_range":
            int(
                is_missing(
                    salary_range
                )
            ),

        "missing_department":
            int(
                is_missing(
                    department
                )
            ),

        "missing_employment_type":
            int(
                is_missing(
                    employment_type
                )
            ),

        "missing_required_experience":
            int(
                is_missing(
                    required_experience
                )
            ),

        "missing_required_education":
            int(
                is_missing(
                    required_education
                )
            ),

        "missing_industry":
            int(
                is_missing(
                    industry
                )
            ),

        "missing_function":
            int(
                is_missing(
                    function
                )
            ),


        # Salary availability
        "has_salary":
            int(
                not is_missing(
                    salary_range
                )
            ),


        # Text/contact signals
        "has_url":
            has_url,

        "has_email":
            has_email,

        "has_phone_like":
            has_phone_like,

        "has_html":
            has_html,

        "has_currency_symbol":
            has_currency_symbol,

        "exclamation_count":
            raw_text.count("!"),

        "all_caps_word_count":
            count_all_caps_words(
                raw_text
            ),


        # Text lengths
        "title_char_length":
            chars(
                title_clean
            ),

        "title_word_length":
            words(
                title_clean
            ),

        "company_profile_char_length":
            chars(
                company_profile_clean
            ),

        "company_profile_word_length":
            words(
                company_profile_clean
            ),

        "description_char_length":
            chars(
                description_clean
            ),

        "description_word_length":
            words(
                description_clean
            ),

        "requirements_char_length":
            chars(
                requirements_clean
            ),

        "requirements_word_length":
            words(
                requirements_clean
            ),

        "benefits_char_length":
            chars(
                benefits_clean
            ),

        "benefits_word_length":
            words(
                benefits_clean
            ),


        # Categoricals
        "country":
            country,

        "employment_type":
            categorical_value(
                employment_type
            ),

        "required_experience":
            categorical_value(
                required_experience
            ),

        "required_education":
            categorical_value(
                required_education
            ),

        "industry":
            categorical_value(
                industry
            ),

        "function":
            categorical_value(
                function
            ),
    }


    return pd.DataFrame([row])