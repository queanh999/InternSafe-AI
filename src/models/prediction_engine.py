from pathlib import Path
import sys

import joblib


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.utils.feature_builder import build_features


# ============================================================
# INTERNSAFE AI - PRODUCTION PREDICTION ENGINE
# ============================================================


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "calibrated"
    / "calibrated_combined_svm.joblib"
)


SCREENING_THRESHOLD = 0.12
HIGH_RISK_THRESHOLD = 0.52


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    probability
):

    if probability >= HIGH_RISK_THRESHOLD:

        return {
            "code": "HIGH",
            "label": "Rủi ro cao",
            "message":
                "Tin tuyển dụng có nhiều đặc điểm "
                "tương đồng với nhóm tin gian lận "
                "trong dữ liệu huấn luyện."
        }

    if probability >= SCREENING_THRESHOLD:

        return {
            "code": "REVIEW",
            "label": "Cần kiểm tra thêm",
            "message":
                "Tin tuyển dụng có một số dấu hiệu "
                "cần được xác minh thêm trước khi "
                "ứng tuyển hoặc cung cấp thông tin cá nhân."
        }

    return {
        "code": "LOW",
        "label": "Rủi ro thấp",
        "message":
            "Hệ thống chưa phát hiện đủ tín hiệu "
            "để đưa tin tuyển dụng vào nhóm cảnh báo."
    }


# ============================================================
# USER-FRIENDLY RULE-BASED SUPPORTING REASONS
#
# IMPORTANT:
# These are NOT independent fraud rules.
# They summarize observable features used by the ML pipeline.
# ============================================================

def build_readable_factors(
    features
):

    row = features.iloc[0]

    risk_factors = []
    safety_factors = []


    # --------------------------------------------------------
    # Company profile
    # --------------------------------------------------------

    if (
        row[
            "missing_company_profile"
        ] == 1
    ):

        risk_factors.append(
            "Không có phần giới thiệu doanh nghiệp."
        )

    else:

        safety_factors.append(
            "Có thông tin giới thiệu doanh nghiệp."
        )


    # --------------------------------------------------------
    # Company logo
    # --------------------------------------------------------

    if (
        row[
            "has_company_logo"
        ] == 0
    ):

        risk_factors.append(
            "Không có thông tin logo doanh nghiệp "
            "trong dữ liệu tin tuyển dụng."
        )

    else:

        safety_factors.append(
            "Tin có thông tin logo doanh nghiệp."
        )


    # --------------------------------------------------------
    # Requirements
    # --------------------------------------------------------

    if (
        row[
            "missing_requirements"
        ] == 1
    ):

        risk_factors.append(
            "Không có phần yêu cầu công việc rõ ràng."
        )

    else:

        safety_factors.append(
            "Có phần yêu cầu công việc."
        )


    # --------------------------------------------------------
    # Experience
    # --------------------------------------------------------

    if (
        row[
            "missing_required_experience"
        ] == 1
    ):

        risk_factors.append(
            "Không nêu yêu cầu kinh nghiệm."
        )


    # --------------------------------------------------------
    # Education
    # --------------------------------------------------------

    if (
        row[
            "missing_required_education"
        ] == 1
    ):

        risk_factors.append(
            "Không nêu yêu cầu học vấn."
        )


    # --------------------------------------------------------
    # Salary
    # --------------------------------------------------------

    if (
        row[
            "has_salary"
        ] == 1
    ):

        safety_factors.append(
            "Tin có cung cấp thông tin mức lương."
        )


    # --------------------------------------------------------
    # Contact signals
    # --------------------------------------------------------

    if (
        row[
            "has_phone_like"
        ] == 1
    ):

        risk_factors.append(
            "Nội dung chứa số điện thoại trực tiếp; "
            "nên xác minh danh tính người tuyển dụng."
        )


    if (
        row[
            "has_email"
        ] == 1
    ):

        risk_factors.append(
            "Nội dung chứa địa chỉ email trực tiếp; "
            "nên kiểm tra tên miền và doanh nghiệp."
        )


    if (
        row[
            "has_url"
        ] == 1
    ):

        risk_factors.append(
            "Tin chứa liên kết bên ngoài; "
            "nên kiểm tra tên miền trước khi truy cập."
        )


    # --------------------------------------------------------
    # Remote
    # --------------------------------------------------------

    if (
        row[
            "telecommuting"
        ] == 1
    ):

        risk_factors.append(
            "Tin cho phép làm việc từ xa; "
            "cần xác minh doanh nghiệp trước khi "
            "chia sẻ thông tin cá nhân."
        )


    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    if (
        row[
            "has_questions"
        ] == 1
    ):

        safety_factors.append(
            "Tin có câu hỏi sàng lọc ứng viên."
        )


    return (
        risk_factors[:5],
        safety_factors[:5]
    )


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def analyze_job(job):

    features = build_features(
        job
    )


    probability = float(
        model.predict_proba(
            features
        )[0, 1]
    )


    risk = get_risk_level(
        probability
    )


    (
        risk_factors,
        safety_factors
    ) = build_readable_factors(
        features
    )


    return {

        "fraud_probability":
            round(
                probability,
                4
            ),

        "risk_percent":
            round(
                probability * 100,
                2
            ),

        "risk_level":
            risk["code"],

        "risk_label":
            risk["label"],

        "message":
            risk["message"],

        "risk_factors":
            risk_factors,

        "safety_factors":
            safety_factors,

        "disclaimer":
            (
                "Kết quả là công cụ hỗ trợ đánh giá rủi ro "
                "dựa trên mô hình học máy, không phải kết luận "
                "pháp lý rằng một cá nhân hoặc doanh nghiệp "
                "thực hiện hành vi gian lận."
            )
    }


# ============================================================
# TEST WHEN RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    example_job = {

        "title":
            "Data Entry Assistant",

        "company_profile":
            "",

        "description":
            (
                "Work from home. "
                "We are looking for data entry staff. "
                "Apply using the link below."
            ),

        "requirements":
            "",

        "benefits":
            "",

        "location":
            "US, NY, New York",

        "department":
            "",

        "salary_range":
            "",

        "employment_type":
            "Part-time",

        "required_experience":
            "Not Applicable",

        "required_education":
            "High School or equivalent",

        "industry":
            "Accounting",

        "function":
            "Administrative",

        "telecommuting":
            1,

        "has_company_logo":
            0,

        "has_questions":
            0
    }


    result = analyze_job(
        example_job
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "INTERNSAFE AI - PREDICTION"
    )

    print(
        "=" * 70
    )


    for key, value in (
        result.items()
    ):

        print(
            f"\n{key}:"
        )

        print(
            value
        )