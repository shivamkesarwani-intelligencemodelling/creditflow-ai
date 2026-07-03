from dataclasses import dataclass

import pandas as pd

from app.core import logger


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str]
    warnings: list[str]


class DataValidator:
    def validate(
        self,
        df: pd.DataFrame,
        required_columns: list[str] | None = None,
    ) -> ValidationResult:
        errors = []
        warnings = []

        if df.empty:
            errors.append("Dataset is empty")

        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]

            if missing:
                errors.append(f"Missing required columns: {missing}")

        duplicates = df.duplicated().sum()

        if duplicates > 0:
            warnings.append(f"{duplicates} duplicate rows detected")

        null_percentage = (df.isnull().mean() * 100).round(2)

        for column, pct in null_percentage.items():
            if pct > 50:
                warnings.append(f"{column}: {pct}% missing values")

        passed = len(errors) == 0

        if passed:
            logger.info("Dataset validation passed")

        else:
            logger.error("Dataset validation failed")

        return ValidationResult(
            passed=passed,
            errors=errors,
            warnings=warnings,
        )
