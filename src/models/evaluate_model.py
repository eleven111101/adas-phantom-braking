from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds)
    }
    return metrics