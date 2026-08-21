# InternSafe AI - Project Documentation


# 1. Problem Statement

Online recruitment fraud is becoming increasingly common.
Many fraudulent job postings use:

- Fake company information
- Attractive remote jobs
- Missing employer details
- Suspicious descriptions
- Unrealistic requirements


InternSafe AI aims to automatically detect risky job postings using Machine Learning.


---

# 2. System Overview


The system contains four main layers:


## Layer 1: User Interface

Technology:

- HTML
- CSS
- JavaScript


Functions:

- Input job information
- Display risk level
- Show explanations


---

## Layer 2: Backend API

Technology:

- FastAPI


Responsibilities:

- Receive user input
- Process data
- Run ML inference
- Return prediction


---

## Layer 3: Machine Learning Pipeline


Input:

Job posting information


Features:

### Text Features

- Job title
- Description
- Requirements
- Benefits


Technique:

TF-IDF Vectorization


### Metadata Features

- Company logo availability
- Salary information
- Employment type
- Education
- Experience


---

## Layer 4: Prediction Engine


Model:

Calibrated Linear SVM


Output:

- Fraud probability
- Risk level
- Explanation factors


---

# 3. Machine Learning Workflow


Raw Dataset

↓

Data Exploration

↓

Data Cleaning

↓

Feature Engineering

↓

Model Training

↓

Hyperparameter Optimization

↓

Calibration

↓

Evaluation

↓

Deployment



# 4. Explainable AI


InternSafe AI does not only predict.

It explains:

Why a job is suspicious:

Examples:

- Missing company profile
- Suspicious keywords
- Remote-only position
- Missing requirements


Why a job looks safer:

Examples:

- Company information exists
- Salary provided
- Detailed requirements


---

# 5. Risk Classification


## LOW

Low probability of fraud.


## REVIEW

Requires additional verification.


## HIGH

Strong similarity with fraudulent patterns.



# 6. Deployment


Platform:

GitHub Codespaces


Backend:

FastAPI


Model:

Scikit-learn

