# Relatório — Parte 2: Classificação de Imagens Médicas com CNN

**Projeto:** CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
**Autor:** Carlos Mário Vieira de Melo Filho — RM563769

## 1. Objetivo

Classificar radiografias de tórax em `NORMAL` vs. `PNEUMONIA` comparando duas
abordagens exigidas pelo enunciado: **CNN treinada do zero** e **Transfer Learning**
com VGG16. Os modelos são salvos em `models/*.keras` e servidos pelo protótipo Flask.

## 2. Abordagem A — CNN do zero

Arquitetura sequencial simples (notebook `parte2`):

```
Input(224×224×3) → Rescaling(1/255) → DataAugmentation
→ [Conv2D(16) → MaxPool] → [Conv2D(32) → MaxPool]
→ [Conv2D(64) → MaxPool] → [Conv2D(64) → MaxPool]
→ GlobalAveragePooling2D → Dropout(0.3) → Dense(64, relu) → Dense(1, sigmoid)
```

- Normalização e augmentation **embutidos no modelo** (1ª camadas) → o `.keras`
  salvo carrega seu próprio pré-processamento e o Flask recebe a imagem crua.
- Otimizador Adam, perda `binary_crossentropy`.

## 3. Abordagem B — Transfer Learning (VGG16)

- Base **VGG16** pré-treinada (ImageNet), `include_top=False`, **congelada**.
- Ligada via `input_tensor` (grafo **plano**) → habilita o Grad-CAM a acessar
  diretamente `block5_conv3` no app.
- Cabeça nova: `GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid)`.

> Nota técnica: usamos `Rescaling(1/255)` também no ramo VGG16 (em vez do
> `vgg16.preprocess_input` caffe/BGR) para manter **um único pré-processamento**
> entre os dois modelos e o Flask. A cabeça treinável adapta-se às features; é uma
> simplificação consciente, adequada ao escopo do protótipo.

## 4. Avaliação

Métricas no conjunto de **teste** (`sklearn`): acurácia, matriz de confusão,
precisão, recall e F1-score. Geradas pelo notebook e salvas em `docs/`:

- `parte2_curvas_scratch.png`, `parte2_curvas_transfer.png` — curvas de treino.
- `parte2_matriz_scratch.png`, `parte2_matriz_transfer.png` — matrizes de confusão.
- `parte2_comparacao.png` — comparação das métricas.

### Resultados

Conjunto de teste (200 imagens, 100 por classe). Precisão/Recall/F1 referentes à
classe positiva `PNEUMONIA`. Execução local em CPU — CNN do zero: 8 épocas;
Transfer Learning: 5 épocas.

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| CNN do zero | 0,740 | 1,000 | 0,480 | 0,649 |
| Transfer (VGG16) | **0,855** | 0,938 | **0,760** | **0,840** |

**Leitura clínica:** a CNN do zero ficou conservadora — quando aponta pneumonia
acerta (precisão 1,0), mas deixa passar mais da metade dos casos (recall 0,48),
o que é perigoso em triagem (muitos falsos negativos). O **Transfer Learning**
equilibra melhor (recall 0,76 e F1 0,84), confirmando o ganho de reaproveitar as
features do ImageNet num subset pequeno. Valores podem variar entre execuções
(inicialização aleatória); aumentar épocas/dados tende a elevar ainda mais o recall.

## 5. Protótipo de apresentação (Flask)

App em `webapp/` (`make web`): upload de uma radiografia → escolha do modelo →
exibe **classe, confiança e mapa de calor Grad-CAM**. O Grad-CAM destaca as regiões
que mais pesaram na decisão, dando **interpretabilidade** (apoio à triagem, não
diagnóstico).

## 6. Limitações e responsabilidade

- Subset pequeno → métricas indicativas, não generalizáveis para uso clínico.
- Possíveis vieses de aquisição (equipamento/população) herdados do dataset.
- Recomenda-se, em produção: dataset maior, validação externa, calibração de
  limiar, monitoramento de *drift* e conformidade com LGPD/normas médicas
  (alinhado à disciplina de Governança da Fase 4).
