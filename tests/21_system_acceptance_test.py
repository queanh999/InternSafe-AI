from pathlib import Path
import sys

from fastapi.testclient import TestClient


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.backend.main import app


# ============================================================
# TEST CLIENT
# ============================================================

client = TestClient(app)


passed = 0
failed = 0


def check(name, condition, details=""):

    global passed, failed

    if condition:

        passed += 1
        print(f"[PASS] {name}")

    else:

        failed += 1
        print(f"[FAIL] {name}")

        if details:
            print(f"       {details}")


# ============================================================
# 1. WEBSITE
# ============================================================

print("\n" + "=" * 85)
print("1. FRONTEND")
print("=" * 85)

response = client.get("/")

check(
    "Trang chủ trả HTTP 200",
    response.status_code == 200,
    str(response.status_code)
)

check(
    "Trang chủ trả HTML",
    "InternSafe AI" in response.text
)


# ============================================================
# 2. STATIC FILES
# ============================================================

print("\n" + "=" * 85)
print("2. STATIC FILES")
print("=" * 85)

css_response = client.get(
    "/static/style.css"
)

js_response = client.get(
    "/static/script.js"
)

check(
    "style.css tải được",
    css_response.status_code == 200
)

check(
    "script.js tải được",
    js_response.status_code == 200
)


# ============================================================
# 3. HEALTH CHECK
# ============================================================

print("\n" + "=" * 85)
print("3. BACKEND")
print("=" * 85)

response = client.get(
    "/api/health"
)

check(
    "Health endpoint hoạt động",
    response.status_code == 200
)

if response.status_code == 200:

    data = response.json()

    check(
        "Model được khai báo",
        "model" in data
    )


# ============================================================
# 4. DEMO CASES
# ============================================================

print("\n" + "=" * 85)
print("4. DEMO CASES")
print("=" * 85)

response = client.get(
    "/api/demo-cases"
)

check(
    "Demo endpoint hoạt động",
    response.status_code == 200
)

if response.status_code == 200:

    demo_data = response.json()

    check(
        "Có LOW demo",
        "LOW" in demo_data.get(
            "cases",
            {}
        )
    )

    check(
        "Có REVIEW demo",
        "REVIEW" in demo_data.get(
            "cases",
            {}
        )
    )

    check(
        "Có HIGH demo",
        "HIGH" in demo_data.get(
            "cases",
            {}
        )
    )


# ============================================================
# HELPER
# ============================================================

def analyze_case(
    name,
    payload
):

    response = client.post(
        "/api/analyze",
        json=payload
    )

    check(
        f"{name}: API trả HTTP 200",
        response.status_code == 200,
        response.text
    )

    if response.status_code != 200:
        return

    result = response.json()

    check(
        f"{name}: success=true",
        result.get("success") is True
    )

    analysis = result.get(
        "analysis",
        {}
    )

    probability = analysis.get(
        "fraud_probability"
    )

    risk_percent = analysis.get(
        "risk_percent"
    )

    risk_level = analysis.get(
        "risk_level"
    )

    check(
        f"{name}: có xác suất",
        isinstance(
            probability,
            (int, float)
        )
    )

    check(
        f"{name}: xác suất hợp lệ",
        isinstance(
            probability,
            (int, float)
        )
        and
        0 <= probability <= 1
    )

    check(
        f"{name}: Risk Score hợp lệ",
        isinstance(
            risk_percent,
            (int, float)
        )
        and
        0 <= risk_percent <= 100
    )

    check(
        f"{name}: Risk Level hợp lệ",
        risk_level in [
            "LOW",
            "REVIEW",
            "HIGH"
        ]
    )

    check(
        f"{name}: có risk_factors",
        isinstance(
            analysis.get(
                "risk_factors"
            ),
            list
        )
    )

    check(
        f"{name}: có safety_factors",
        isinstance(
            analysis.get(
                "safety_factors"
            ),
            list
        )
    )

    print(
        f"       Risk = "
        f"{risk_percent}% "
        f"({risk_level})"
    )


# ============================================================
# 5. NORMAL VIETNAMESE INPUT
# ============================================================

print("\n" + "=" * 85)
print("5. VIETNAMESE INPUT")
print("=" * 85)

analyze_case(

    "Tin tuyển dụng tiếng Việt",

    {
        "title":
            "Lập trình viên Web",

        "company_profile":
            (
                "Công ty công nghệ phát triển "
                "các giải pháp phần mềm."
            ),

        "description":
            (
                "Tuyển lập trình viên Web "
                "tham gia phát triển sản phẩm."
            ),

        "requirements":
            (
                "Biết HTML, CSS, JavaScript "
                "và có khả năng làm việc nhóm."
            ),

        "benefits":
            (
                "Lương cạnh tranh, bảo hiểm "
                "và đào tạo chuyên môn."
            ),

        "location":
            "VN, Ho Chi Minh City",

        "department":
            "Engineering",

        "salary_range":
            "15000000-25000000",

        "employment_type":
            "Full-time",

        "required_experience":
            "Entry level",

        "required_education":
            "Bachelor's Degree",

        "industry":
            "Computer Software",

        "function":
            "Information Technology",

        "telecommuting":
            0,

        "has_company_logo":
            1,

        "has_questions":
            1
    }
)


# ============================================================
# 6. MINIMAL INPUT
# ============================================================

print("\n" + "=" * 85)
print("6. MINIMAL INPUT")
print("=" * 85)

analyze_case(

    "Chỉ có tiêu đề",

    {
        "title":
            "Nhân viên văn phòng"
    }
)


# ============================================================
# 7. SPECIAL CHARACTERS
# ============================================================

print("\n" + "=" * 85)
print("7. SPECIAL INPUT")
print("=" * 85)

analyze_case(

    "URL / email / phone / HTML",

    {
        "title":
            "Nhân viên hỗ trợ",

        "company_profile":
            "",

        "description":
            (
                "<b>Tuyển gấp!!!</b> "
                "Liên hệ example@gmail.com "
                "hoặc +84 912 345 678. "
                "Xem tại https://example.com"
            ),

        "requirements":
            "",

        "benefits":
            "",

        "location":
            "VN, Hanoi",

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
            "",

        "function":
            "",

        "telecommuting":
            1,

        "has_company_logo":
            0,

        "has_questions":
            0
    }
)


# ============================================================
# 8. VERY LONG DESCRIPTION
# ============================================================

print("\n" + "=" * 85)
print("8. LONG INPUT")
print("=" * 85)

long_description = (
    "Mô tả công việc và yêu cầu ứng viên. "
    * 500
)

analyze_case(

    "Nội dung dài",

    {
        "title":
            "Chuyên viên tuyển dụng",

        "company_profile":
            "Doanh nghiệp hoạt động trong lĩnh vực nhân sự.",

        "description":
            long_description,

        "requirements":
            "Có kỹ năng giao tiếp.",

        "benefits":
            "Bảo hiểm và chế độ nghỉ phép.",

        "location":
            "VN, Hanoi",

        "department":
            "Human Resources",

        "salary_range":
            "",

        "employment_type":
            "Full-time",

        "required_experience":
            "Associate",

        "required_education":
            "Bachelor's Degree",

        "industry":
            "Human Resources",

        "function":
            "Human Resources",

        "telecommuting":
            0,

        "has_company_logo":
            1,

        "has_questions":
            1
    }
)


# ============================================================
# 9. INVALID REQUEST
# ============================================================

print("\n" + "=" * 85)
print("9. VALIDATION")
print("=" * 85)

response = client.post(
    "/api/analyze",
    json={}
)

check(
    "Không có title → HTTP 422",
    response.status_code == 422,
    str(response.status_code)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 85)
print("SYSTEM ACCEPTANCE TEST SUMMARY")
print("=" * 85)

print(
    f"\nPassed : {passed}"
)

print(
    f"Failed : {failed}"
)

print(
    f"Total  : {passed + failed}"
)


if failed == 0:

    print(
        "\nALL SYSTEM TESTS PASSED."
    )

    print(
        "InternSafe AI is ready "
        "for deployment preparation."
    )

else:

    print(
        "\nSome tests failed."
    )

    print(
        "Fix application errors only. "
        "Do NOT retune the ML model "
        "using these tests."
    )