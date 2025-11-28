import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import RESULTS_DIR, FIGURES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        logger.info(f"Lendo métricas de: {metrics_path}")
        df = pd.read_csv(metrics_path)
        logger.info(f"Métricas carregadas: {len(df)} linhas.")
    else:
        logger.info("Arquivo metrics.csv não encontrado, seguindo adiante.")

    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix_svm.csv")
    if not os.path.exists(cm_path):
        logger.info("Matriz de confusão da SVM não encontrada, nenhum gráfico será gerado.")
        return

    logger.info(f"Lendo matriz de confusão da SVM de: {cm_path}")
    cm = pd.read_csv(cm_path).values

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negativo", "Positivo"])
    ax.set_yticklabels(["Negativo", "Positivo"])
    fig.colorbar(im, ax=ax)

    out_path = os.path.join(FIGURES_DIR, "fig_cms.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Figura da matriz de confusão salva em: {out_path}")

if __name__ == "__main__":
    main()
