import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from .config import CLEAN_DATA_PATH, SEED
from .utils import stratified_split, save_metrics_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def build_models():
    nb = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ])
    lr = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED, solver="liblinear"))
    ])
    svm = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf", LinearSVC(random_state=SEED))
    ])
    return {"NB_TFIDF": nb, "LR_TFIDF": lr, "SVM_TFIDF": svm}

def tune_svm(model, X_train, y_train):
    logger.info("Iniciando GridSearchCV para SVM.")
    params = {"clf__C": [0.5, 1.0, 2.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(model, params, cv=cv, n_jobs=-1, scoring="f1")
    gs.fit(X_train, y_train)
    logger.info(f"Melhor parâmetro C para SVM: {gs.best_params_['clf__C']}")
    return gs.best_estimator_

def main():
    logger.info("Iniciando treinamento dos modelos clássicos.")
    df = pd.read_csv(CLEAN_DATA_PATH)
    logger.info(f"Base processada carregada: {len(df)} instâncias.")

    X_train, X_test, y_train, y_test = stratified_split(df)
    logger.info(f"Tamanho treino: {len(X_train)} | teste: {len(X_test)}")

    models = build_models()

    models["SVM_TFIDF"] = tune_svm(models["SVM_TFIDF"], X_train, y_train)

    from .evaluate import evaluate_model
    rows = []
    for name, model in models.items():
        logger.info(f"Treinando modelo: {name}")
        model.fit(X_train, y_train)
        logger.info(f"Avaliando modelo: {name}")
        metrics = evaluate_model(model, X_test, y_test, name)
        logger.info(f"Métricas {name}: {metrics}")
        rows.append(metrics)

    save_metrics_csv(rows)
    logger.info("Métricas dos modelos clássicos salvas em results/metrics.csv.")

if __name__ == "__main__":
    main()
