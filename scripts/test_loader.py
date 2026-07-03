from training.data import DataLoader

loader = DataLoader()

df = loader.load("application_train.csv")

print(df.head())
