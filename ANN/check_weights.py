import joblib

weights = joblib.load("models/weights.pkl")

for i, w in enumerate(weights):
    print(i, w.shape)