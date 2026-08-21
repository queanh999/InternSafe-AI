# InternSafe AI Architecture


## Overview


User

↓

Web Interface

↓

FastAPI Backend

↓

Feature Engineering Pipeline

↓

Machine Learning Model

↓

Risk Prediction + Explanation


## Components


### Frontend

HTML/CSS/JavaScript

Responsible for:

- User input
- Visualization
- Risk display


### Backend

FastAPI

Responsible for:

- API processing
- Feature transformation
- Model inference


### ML Pipeline

Input:

Job posting data


Features:

- TF-IDF text features
- Metadata features
- Missing information signals


Model:

Calibrated Linear SVM


Output:

- Fraud probability
- Risk level
- Explanation factors