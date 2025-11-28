from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from .utils import save_confusion_matrix

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    if name == "SVM_TFIDF":
        cm = confusion_matrix(y_test, preds)
        save_confusion_matrix(cm)
    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4)
    }