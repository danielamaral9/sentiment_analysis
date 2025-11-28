import os
import re
import logging
import pandas as pd
from sklearn.utils import shuffle
from .config import RAW_DATA_PATH, CLEAN_DATA_PATH
from .utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_text(s):
    s = s.lower()
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    logger.info("Iniciando pré-processamento.")
    set_seed()

    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    logger.info(f"Lendo dados brutos de: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Total de linhas lidas: {len(df)}")

    df = df.drop_duplicates()
    df = df.dropna()
    logger.info(f"Após remoção de duplicatas e NaN: {len(df)} linhas")

    logger.info("Aplicando limpeza de texto.")
    df["review"] = df["review"].astype(str).apply(clean_text)

    df = df[df["review"].str.len() > 0]
    logger.info(f"Após remoção de textos vazios: {len(df)} linhas")

    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    logger.info("Mapeando rótulos 'positive/negative' para 1/0.")

    df = shuffle(df, random_state=42).reset_index(drop=True)
    logger.info("Embaralhando linhas.")

    df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info(f"Arquivo processado salvo em: {CLEAN_DATA_PATH}")

if __name__ == "__main__":
    main()