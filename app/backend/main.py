
from pathlib import Path
import sys
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT ML ENGINE
# ============================================================

from src.models.prediction_engine import analyze_job


# ============================================================
# FRONTEND PATHS
# ============================================================

FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
DEMO_CASES_PATH = (
    PROJECT_ROOT
    / "app"
    / "demo_cases"
    / "demo_cases.json"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="InternSafe AI API",
    description="Hệ thống hỗ trợ đánh giá rủi ro tin tuyển dụng bằng Machine Learning.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class JobPostingRequest(BaseModel):

    title: str = Field(..., min_length=1)

    company_profile: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""

    location: str = ""
    department: str = ""
    salary_range: str = ""

    employment_type: str = ""
    required_experience: str = ""
    required_education: str = ""

    industry: str = ""
    function: str = ""

    telecommuting: int = Field(default=0, ge=0, le=1)
    has_company_logo: int = Field(default=0, ge=0, le=1)
    has_questions: int = Field(default=0, ge=0, le=1)


# ============================================================
# WEBSITE
# ============================================================

@app.get("/", include_in_schema=False)
def website():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )


# ============================================================
# API STATUS
# ============================================================

@app.get("/api/status")
def status():

    return {
        "app": "InternSafe AI",
        "status": "running",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",
        "model": "Calibrated Combined Linear SVM"
    }

# ============================================================
# DEMO CASES
# ============================================================

@app.get("/api/demo-cases")
def get_demo_cases():

    try:

        with open(
            DEMO_CASES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            cases = json.load(file)

        return {
            "success": True,
            "cases": cases
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
# ============================================================
# ANALYZE
# ============================================================

@app.post("/api/analyze")
def analyze_job_posting(job: JobPostingRequest):

    try:

        result = analyze_job(
            job.model_dump()
        )

        return {
            "success": True,
            "analysis": result
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )