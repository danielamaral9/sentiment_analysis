import logging
import sys
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from .config import CLEAN_DATA_PATH, SEED, RESULTS_DIR
from .utils import stratified_split, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"


def tokenize_function(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def escolher_dispositivo():
    cuda_disp = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available(): {cuda_disp}")

    if cuda_disp:
        device = "cuda"
        print(f"\nDispositivo detectado para treinamento: {device.upper()}")
        resp = input("Deseja usar a GPU (cuda) para o treinamento? [s/n]: ").strip().lower()
        if resp != "s":
            device = "cpu"
            print("Treinamento será executado na CPU.\n")
        else:
            print("Treinamento será executado na GPU (cuda).\n")
    else:
        device = "cpu"
        print("\nGPU não está disponível (torch.cuda.is_available() == False).")
        print("O treinamento será feito na CPU, o que pode ser bem mais lento.")
        resp = input("Deseja continuar mesmo assim? [s/n]: ").strip().lower()
        if resp != "s":
            print("Execução cancelada pelo usuário.")
            sys.exit(0)
        print("Continuando com treinamento na CPU.\n")

    logger.info(f"Dispositivo escolhido para treinamento: {device}")
    return device


def main():
    logger.info("Iniciando treinamento do DistilBERT.")
    df = pd.read_csv(CLEAN_DATA_PATH)
    logger.info(f"Base processada carregada: {len(df)} instâncias.")

    X_train, X_test, y_train, y_test = stratified_split(df)
    logger.info(f"Tamanho treino: {len(X_train)} | teste: {len(X_test)}")

    train_ds = Dataset.from_dict(
        {"text": X_train.tolist(), "label": y_train.tolist()}
    )
    test_ds = Dataset.from_dict(
        {"text": X_test.tolist(), "label": y_test.tolist()}
    )

    logger.info("Carregando tokenizer e modelo DistilBERT.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )
    test_ds = test_ds.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True,
    )

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    test_ds.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    logger.info(f"Exemplo de chaves no train_ds: {train_ds[0].keys()}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    device = escolher_dispositivo()
    # Trainer já move para o device certo, mas manter isso não atrapalha
    model.to(device)

    args = TrainingArguments(
        output_dir=f"{RESULTS_DIR}/distilbert",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        seed=SEED,
        fp16=False,
        bf16=False,  # para evitar surpresas; se quiser, pode ativar depois
        report_to="none",
        remove_unused_columns=False,  # evita problemas com inspeção de assinatura
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Iniciando treinamento do DistilBERT (3 épocas).")
    trainer.train()
    logger.info("Treinamento concluído. Iniciando avaliação.")
    metrics = trainer.evaluate()
    logger.info(f"Métricas DistilBERT: {metrics}")

    save_metrics({"DISTILBERT": metrics})
    logger.info("Métricas do DistilBERT salvas em results/metrics.json.")


if __name__ == "__main__":
    main()
