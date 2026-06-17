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
| **Ir Além 1** | Ética e Governança: vieses do dataset + métricas de *fairness* | `notebooks/ir_alem1_fairness.ipynb`, `src/fairness.py`, `docs/RELATORIO_IR_ALEM1.md` |
| **Ir Além 2** | App mobile (React Native + Expo) consumindo a API Flask | `mobile/`, API JSON em `webapp/app.py` (`/api/predict`) |

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
src/          utilidades compartilhadas (config, preprocessing, inference, gradcam, fairness)
notebooks/    parte1, parte2 e ir_alem1_fairness
scripts/      download_dataset.py (baixa o dataset e gera o subset balanceado)
webapp/       app Flask: interface web + API JSON (/api/predict) para o mobile
mobile/       app React Native + Expo (Ir Além 2)
models/       modelos .keras gerados pelos notebooks (não versionados)
data/         dataset e subset (não versionados)
docs/         relatórios (Parte 1, Parte 2, Ir Além 1), figuras e CONCEITOS.md
tests/        testes pytest (preprocessing, inference, fairness)
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

## Ir Além

**Ir Além 1 — Ética e Governança (fairness).** Análise de vieses do dataset e
equidade do modelo por subtipo de pneumonia (bacteriana × viral):

```bash
make jupyter   # rode notebooks/ir_alem1_fairness.ipynb
```
Relatório em [`docs/RELATORIO_IR_ALEM1.md`](docs/RELATORIO_IR_ALEM1.md); métricas em
`src/fairness.py` (testadas em `tests/test_fairness.py`).

**Ir Além 2 — App mobile (React Native + Expo).** Consome a API JSON do Flask:

```bash
make web                       # 1) backend em 0.0.0.0:5000
cd mobile && npm install && npx expo start   # 2) app (ver mobile/README.md)
```
A API expõe `GET /api/health` e `POST /api/predict` (multipart `imagem` + `modelo`),
retornando classe, confiança e Grad-CAM em base64. Detalhes em
[`mobile/README.md`](mobile/README.md); roteiro do vídeo em
[`mobile/ROTEIRO_VIDEO.md`](mobile/ROTEIRO_VIDEO.md).

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
