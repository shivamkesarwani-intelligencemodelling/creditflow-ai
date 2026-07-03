from training.data import DataLoader
from training.preprocessing.validate import DataValidator

loader = DataLoader()

df = loader.load("application_train.csv")

validator = DataValidator()

result = validator.validate(
    df,
    required_columns=["TARGET"],
)

print(result)
