# CreditFlow AI

An end-to-end MLOps platform for credit default risk prediction using the Home Credit Default Risk dataset. The project demonstrates modern machine learning engineering practices, including data preprocessing, model training, experiment tracking, explainability, monitoring, and production deployment.

---

## Features

- FastAPI REST API for real-time credit risk prediction
- End-to-end data preprocessing pipeline
- Multiple baseline ML models
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model Factory pattern for interchangeable models
- MLflow experiment tracking with SQLite backend
- Structured logging
- Dockerized development environment
- Automated testing with Pytest
- Code quality with Ruff, Black, and pre-commit
- Production-ready project architecture

---

## Project Structure

```text
creditflow-ai/
│
├── app/
│   ├── api/
│   ├── config/
│   ├── inference/
│   ├── services/
│   ├── utils/
│   └── main.py
│
├── training/
│   ├── data/
│   ├── feature_engineering/
│   ├── evaluation/
│   ├── models/
│   └── pipelines/
│
├── tests/
├── data/
├── docker/
├── airflow/
├── mlflow/
├── dashboards/
└── docs/
```

---

## System Architecture

```text
                Client
                   │
                   ▼
            FastAPI REST API
                   │
      ┌────────────┼─────────────┐
      │            │             │
      ▼            ▼             ▼
Feature Store   Inference     Explainability
(PostgreSQL)     Service          (SHAP)
      │
      ▼
 Monitoring & Logging

──────────────────────────────────────────

Training Pipeline
        │
        ▼
 Feature Engineering
        │
        ▼
 Model Factory
        │
        ├──────────────┐
        │              │
 Logistic Regression   Random Forest
                       │
                       ▼
                    XGBoost
        │
        ▼
 Model Evaluation
        │
        ▼
 MLflow Tracking
        │
        ▼
 Model Registry
```

---

## Technology Stack

| Layer | Technology |
|--------|------------|
| Language | Python 3.12 |
| API | FastAPI |
| ML | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Containerization | Docker |
| Testing | Pytest |
| Formatting | Ruff, Black |
| Package Management | uv |

---

## Implemented Models

| Model | Status |
|--------|--------|
| Logistic Regression | ✅ |
| Random Forest | ✅ |
| XGBoost | ✅ |

---

## Current Metrics

| Model | ROC-AUC | F1 Score |
|--------|---------|----------|
| Logistic Regression | *(update after latest run)* | *(update)* |
| Random Forest | 0.7147 | 0.0028 |
| XGBoost | **0.7611** | **0.0439** |

---

## Running the Project

### Install

```bash
uv sync
```

### Run the API

```bash
uv run uvicorn app.main:app --reload
```

### Train a Model

```bash
uv run python -m training.pipelines.train --model logistic_regression
```

```bash
uv run python -m training.pipelines.train --model random_forest
```

```bash
uv run python -m training.pipelines.train --model xgboost
```

### Run Tests

```bash
uv run pytest
```

### Launch MLflow

```bash
uv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0 \
    --port 5000
```

---

## Roadmap

### Milestone 1
- [x] Backend setup
- [x] Docker
- [x] Testing
- [x] CI-ready structure

### Milestone 2
- [x] Configuration management
- [x] Data ingestion
- [x] Dataset validation
- [x] Feature engineering

### Milestone 3
- [x] Logistic Regression
- [x] Random Forest
- [x] XGBoost
- [x] Model Factory
- [x] MLflow integration
- [ ] Hyperparameter tuning
- [ ] Model benchmarking

### Milestone 4
- [ ] SHAP Explainability
- [ ] Feature importance
- [ ] Prediction explanations

### Milestone 5
- [ ] FastAPI inference service
- [ ] Model registry
- [ ] Monitoring
- [ ] Prometheus metrics
- [ ] Grafana dashboards

---

## License

MIT License
