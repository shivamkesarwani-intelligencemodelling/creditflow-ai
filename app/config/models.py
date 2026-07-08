from pydantic import BaseModel


class LogisticRegressionConfig(BaseModel):
    max_iter: int = 1000
    random_state: int = 42


class RandomForestConfig(BaseModel):
    n_estimators: int = 300
    max_depth: int = 10
    random_state: int = 42
    class_weight: str = "balanced"
    n_jobs: int = -1


class XGBoostConfig(BaseModel):
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    eval_metric: str = "logloss"
    random_state: int = 42
    objective: str = "binary:logistic"
    n_jobs: int = -1
