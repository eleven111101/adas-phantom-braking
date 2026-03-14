import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df, target_column):
    df = df.copy()
    encoder = LabelEncoder()
    df["object_type"] = encoder.fit_transform(df["object_type"])
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y