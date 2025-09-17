import pickle

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

print("Model Loaded:", type(model).__name__)
