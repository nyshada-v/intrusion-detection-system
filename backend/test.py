import joblib

cols = joblib.load("models/feature_cols.pkl")

print(len(cols))
print(cols)