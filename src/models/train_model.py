from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_model(X, y, config):
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model_name = config["model"]["selected_model"]
    model_params = config["models"]
    if model_name == "random_forest":
        params = model_params["random_forest"]
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"]
        )
    elif model_name == "xgboost":
        params = model_params["xgboost"]
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"]
        )
    elif model_name == "logistic_regression":
        params = model_params["logistic_regression"]
        model = LogisticRegression(
            max_iter=params["max_iter"]
        )
    else:
        raise ValueError("Unsupported model selected")
    model.fit(X_train, y_train)
    return model, X_test, y_test