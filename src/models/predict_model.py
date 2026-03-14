import pandas as pd
import joblib

def load_model(path):
    return joblib.load(path)

def predict(model, input_data):
    df = pd.DataFrame([input_data])
    probability = model.predict_proba(df)[0][1]

    return probability