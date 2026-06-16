# CardioIA — Fase 4 / Cap 1: Visão Computacional na Clínica

Protótipo de **Assistente Cardiológico Virtual com Visão Computacional**. O sistema
pré-processa imagens médicas (raios-X de tórax), treina e avalia **CNNs** para
classificar entre `NORMAL` e `PNEUMONIA`, e apresenta o resultado em uma **interface
web Flask** com mapa de calor **Grad-CAM** (interpretabilidade da decisão).

> ⚠️ **Aviso:** projeto **acadêmico** de apoio à decisão. Não é diagnóstico médico
> nem substitui avaliação profissional. Dados de imagem são de dataset público de
> pesquisa (uso educacional).

## Escopo entregue

| Parte | Conteúdo | Onde |
|---|---|---|
| **Parte 1** | Pipeline de pré-processamento e organização das imagens | `notebooks/parte1_preprocessamento.ipynb`, `src/preprocessing.py`, `docs/RELATORIO_PARTE1.md` |
| **Parte 2** | CNN do zero + Transfer Learning (VGG16), avaliação e protótipo | `notebooks/parte2_cnn_classificacao.ipynb`, `webapp/`, `docs/RELATORIO_PARTE2.md` |

📚 Explicação didática de **todos os conceitos** de ML/redes neurais usados (com
ponteiros para o código): [`docs/CONCEITOS.md`](docs/CONCEITOS.md).

## Stack

- **Python 3.11** (TensorFlow não suporta o 3.14 do sistema — venv dedicado).
- **TensorFlow/Keras** — CNNs e VGG16 (`keras.applications`).
- **Flask** — protótipo web de apresentação.
- **scikit-learn / matplotlib / seaborn** — métricas e visualizações.
- **Dataset:** [Chest X-Ray Pneumonia (Kermany et al.)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## Estrutura

```
src/          utilidades compartilhadas (config, preprocessing, inference, gradcam)
notebooks/    parte1 (pré-processamento) e parte2 (CNNs + avaliação)
scripts/      download_dataset.py (baixa o dataset e gera o subset balanceado)
webapp/       app Flask (upload → classificação + confiança + Grad-CAM)
models/       modelos .keras gerados pelos notebooks (não versionados)
data/         dataset e subset (não versionados)
docs/         relatórios das Partes 1 e 2, figuras de métricas e CONCEITOS.md
tests/        testes pytest (pré-processamento e inferência)
```

## Como executar

Pré-requisito: `python3.11` instalado.

```bash
make prep-venv     # cria o venv (Python 3.11) e instala as dependências
make dataset       # baixa o dataset e gera data/chest_xray_subset (balanceado)
make train         # executa parte1 e parte2 → gera models/*.keras e métricas em docs/
make web           # sobe o app Flask em http://localhost:5000
make test          # roda a suíte de testes
```

> O `make dataset` usa `kagglehub` (pode pedir autenticação Kaggle na 1ª vez). Sem
> acesso ao Kaggle, baixe o dataset manualmente e aponte a pasta:
> `export CHEST_XRAY_DIR=/caminho/para/chest_xray` antes de `make dataset`.

Para abrir os notebooks interativamente: `make jupyter`.

## Decisões de projeto

- **Pré-processamento único** (`src/preprocessing.py`) usado por notebooks e Flask,
  evitando divergência treino/inferência. As imagens saem em RGB `224×224` [0,255];
  a normalização (`Rescaling 1/255`) é embutida **dentro de cada modelo**.
- **Transfer Learning em grafo plano** (VGG16 via `input_tensor`) para que o Grad-CAM
  acesse diretamente a última camada convolucional.
- **Subset balanceado e reduzido** para viabilizar treino local em CPU; tamanhos
  configuráveis em `scripts/download_dataset.py`.

## Ética, governança e LGPD

- Dataset público de pesquisa, sem dados pessoais identificáveis; uso estritamente
  educacional.
- A interface deixa explícito que é **apoio à decisão**, não diagnóstico.
- O Grad-CAM fornece **rastreabilidade** da predição (onde o modelo olhou),
  alinhado às boas práticas de IA confiável e auditável.

## Equipe

> Atualize com os integrantes do grupo (2 a 5) para o ponto extra de trabalho em equipe.

| Nome | RM |
|---|---|
| Carlos Mário Vieira de Melo Filho | RM563769 |
