# 📘 Comparação entre Modelos Clássicos e Transformers na Análise de Sentimentos

Este repositório contém o código-fonte, scripts de pré-processamento e experimentos utilizados no trabalho **"Comparação entre Modelos Clássicos e Transformers na Análise de Sentimentos"**.  
O projeto compara modelos clássicos de Aprendizado de Máquina (Naive Bayes, Regressão Logística e SVM) com o modelo **DistilBERT**, utilizando o dataset **IMDb Movie Reviews**.

O objetivo é avaliar desempenho, custo computacional e potencial de generalização entre abordagens tradicionais baseadas em TF-IDF e modelos de linguagem baseados em Transformers.

---

## 📂 Estrutura do Repositório

```
.
├── data/
│   ├── raw/              # Dataset original (não incluído no repositório)
│   └── processed/        # Dataset limpo e balanceado (gerado pelos scripts)
├── src/
│   ├── config.py         # Caminhos, seeds e constantes
│   ├── utils.py          # Funções auxiliares (seed, split, salvar métricas)
│   ├── preprocess.py     # Limpeza e preparação da base
│   ├── train_classical.py
│   └── train_distilbert.py
├── results/
│   ├── metrics.json
│   └── ...
├── figures/
├── requirements.txt
└── README.md
```

---

## 📥 Download do Dataset IMDb

O dataset **não está incluído** no repositório devido ao tamanho.  
Você deve baixá-lo manualmente e colocá-lo em:

```
data/raw/IMDB_Dataset.csv
```

### Link oficial para download:

- Kaggle (recomendado):  
  https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

## ⚙️ Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/danielamaral9/sentiment-analysis
cd sentiment-analysis
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Linux/Mac
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

---

## 🚀 Execução dos Experimentos

### 1. Pré-processar o dataset

```bash
python -m src.preprocess
```

### 2. Treinar modelos clássicos

```bash
python -m src.train_classical
```

### 3. Treinar o DistilBERT

```bash
python -m src.train_distilbert
```

---

## 📊 Resultados Esperados

| Modelo        | F1-Score aproximado |
|--------------|----------------------|
| Naive Bayes  | ~0.89 |
| LogReg TF-IDF| ~0.91 |
| **SVM TF-IDF** | **~0.92** |
| DistilBERT   | ~0.88–0.90 |

---

## 🧪 Reprodutibilidade

O projeto adota:

- seeds globais (`SEED = 42`)
- scripts independentes
- separação entre dados brutos e processados
- métricas salvas automaticamente

---

## 📜 Como citar

```
Ribeiro Amaral, Daniel (2025). 
Comparação entre Modelos Clássicos e Transformers na Análise de Sentimentos.
Repositório GitHub: https://github.com/danielamaral9/sentiment-analysis
```

---

## 🤝 Contribuições

Pull requests e Issues são bem-vindos.

---
