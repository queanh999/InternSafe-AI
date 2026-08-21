# System Architecture


```mermaid
flowchart TD

A[User]

B[Web Interface]

C[FastAPI Backend]

D[Feature Engineering]

E[Machine Learning Model]

F[Risk Prediction]

G[Explainable AI]


A --> B

B --> C

C --> D

D --> E

E --> F

F --> G

G --> B