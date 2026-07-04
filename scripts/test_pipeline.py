from training.data import DataLoader
from training.feature_engineering.features import get_feature_types
from training.feature_engineering.pipeline import build_preprocessor

loader = DataLoader()

df = loader.load("application_train.csv")

# Remove target before preprocessing
X = df.drop(columns=["TARGET"])

num_features, cat_features = get_feature_types(X)

preprocessor = build_preprocessor(
    num_features,
    cat_features,
)

X_processed = preprocessor.fit_transform(X)

print(X_processed.shape)
