# Relatório — Parte 1: Pré-processamento e Organização das Imagens

**Projeto:** CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
**Autor:** Carlos Mário Vieira de Melo Filho — RM563769

## 1. Dataset escolhido

**Chest X-Ray Pneumonia** (Kermany et al., 2018), disponível no Kaggle
(`paultimothymooney/chest-xray-pneumonia`). São radiografias de tórax rotuladas
em duas classes — `NORMAL` e `PNEUMONIA` — adequadas a uma tarefa de
**classificação binária** de apoio à triagem clínica.

**Justificativa:** é um dataset público consolidado, com tamanho gerenciável,
rótulos confiáveis e já organizado por classe — ideal para demonstrar tanto uma
CNN treinada do zero quanto Transfer Learning, dentro do tempo de uma CPU local.

## 2. Subset balanceado

O dataset original é desbalanceado (mais casos de pneumonia) e o split de
validação oficial é muito pequeno. Para um treino local reprodutível, o script
`scripts/download_dataset.py` reagrupa as imagens e gera um **subset balanceado**:

| Split | NORMAL | PNEUMONIA |
|---|---|---|
| train | 400 | 400 |
| val | 100 | 100 |
| test | 100 | 100 |

(Quantidades configuráveis via `--train/--val/--test`.) O balanceamento evita que
a acurácia seja inflada pela classe majoritária e torna as métricas mais honestas.

## 3. Pipeline de pré-processamento

Implementado em `src/preprocessing.py` (fonte única, reutilizada pelo app Flask):

1. **Leitura por pasta** com `image_dataset_from_directory`; o rótulo é inferido do
   nome da pasta (`NORMAL`=0, `PNEUMONIA`=1, ordem fixada por `class_names`).
2. **Conversão para RGB** — radiografias são monocromáticas; replicamos para 3
   canais porque o VGG16 (Parte 2) espera entrada RGB.
3. **Redimensionamento para 224×224** — entrada padrão do VGG16 e suficiente para a
   CNN do zero, padronizando o tamanho variável das imagens originais.
4. **Batches + `prefetch`** — eficiência de I/O durante o treino.
5. **Normalização (escala [0,1])** — implementada como **primeira camada do modelo**
   (`Rescaling 1/255`), e não no dataset. Decisão deliberada para garantir que treino
   e inferência (Flask) apliquem exatamente a mesma transformação, eliminando o risco
   de *train/serve skew*.
6. **Data augmentation** (treino apenas) — flip horizontal, rotação e zoom suaves.
   Evita-se flip vertical para preservar a orientação clínica da radiografia.

## 4. Conjuntos de treino, validação e teste

- **Treino:** ajuste dos pesos.
- **Validação:** monitoramento por época (early signal de overfitting).
- **Teste:** avaliação final em dados nunca vistos (Parte 2).

## 5. Artefatos gerados

Figuras salvas em `docs/` pelo notebook `parte1_preprocessamento.ipynb`:
`parte1_distribuicao_classes.png`, `parte1_amostras.png`, `parte1_augmentation.png`.

## 6. Considerações de responsabilidade

Dados públicos de pesquisa, sem identificadores pessoais; uso educacional. O
pré-processamento não introduz vieses de aquisição além dos já presentes no dataset
(equipamento, população) — limitação documentada para a etapa de avaliação.
