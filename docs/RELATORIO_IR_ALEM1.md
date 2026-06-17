# Ir Além 1 — Ética e Governança em Visão Computacional

**Projeto:** CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
**Autor:** Carlos Mário Vieira de Melo Filho — RM563769
**Notebook de apoio:** `notebooks/ir_alem1_fairness.ipynb`

## 1. Objetivo

Analisar vieses do dataset de raios-X (Kermany) e do modelo treinado, aplicar
métricas de **fairness** e propor práticas de mitigação, na ótica de uma IA
**confiável, auditável e responsável** (alinhada à disciplina de Governança).

## 2. Limitações do dataset

**2.1 Desbalanceamento.** O dataset original é desbalanceado (~3:1 a favor de
PNEUMONIA) e o split de validação oficial tem apenas **16 imagens** — insuficiente
para estimar desempenho de forma estável. *Mitigação adotada:* construímos um subset
**balanceado** 50/50 (NORMAL/PNEUMONIA), com val/test próprios, para que a acurácia
não seja inflada pela classe majoritária.

**2.2 Representatividade.** As imagens provêm de **pacientes pediátricos (1–5 anos)**
de um **único centro** (Guangzhou, China). Logo, o modelo **não generaliza** para
adultos, idosos, outras populações, equipamentos ou protocolos de aquisição. Usá-lo
fora desse contexto seria um viés de representatividade grave.

**2.3 Viés intra-classe.** Dentro de PNEUMONIA há dois subtipos — **bacteriana** e
**viral** — também desbalanceados no dataset. Como a pneumonia viral tende a
apresentar padrões radiográficos mais difusos e sutis, há risco de o modelo ser
menos sensível a esse subgrupo. Esse é o eixo da nossa análise de fairness.

## 3. Métricas de fairness

Avaliamos o modelo de Transfer Learning (VGG16) no conjunto de teste (200 imagens),
separando a classe PNEUMONIA por subtipo. O critério adequado aqui é **igualdade de
oportunidade** (*equal opportunity*): **paridade de recall** entre os subgrupos — ou
seja, o modelo deve detectar pneumonia viral tão bem quanto a bacteriana.

| Subgrupo | n | Recall (sensibilidade) | FNR (falsos negativos) |
|---|---|---|---|
| Pneumonia **bacteriana** | 58 | **0,810** | 0,190 |
| Pneumonia **viral** | 42 | **0,738** | 0,262 |
| NORMAL | 100 | acurácia 0,94; só 6% marcados como pneumonia | — |

- **Equal Opportunity Difference (EOD)** = 0,810 − 0,738 = **0,072**.
- **Razão de recall** (menor/maior) = 0,738 / 0,810 = **0,911** — acima do limiar
  clássico de 0,80 (*regra dos 80%*).

**Nota metodológica importante:** *não* usamos **demographic parity / disparate
impact** sobre todos os grupos. Isso seria **incorreto** aqui, porque as taxas-base
diferem **legitimamente** por condição clínica — um paciente NORMAL *deve* ter baixa
taxa de "positivo". Forçar paridade de seleção entre NORMAL e PNEUMONIA pioraria o
modelo. O critério clinicamente correto é a paridade de **recall** entre os subtipos
de pneumonia (equal opportunity).

## 4. Interpretação

O modelo **deixa passar mais pneumonias virais** (FNR 26%) do que bacterianas
(FNR 19%). Embora a razão de recall (0,911) esteja dentro do limiar de 80%, a
diferença é **clinicamente relevante**: falsos negativos significam pacientes com
pneumonia recebendo alta indevida. O viés é coerente com (a) o menor número de
exemplos virais no treino e (b) a natureza mais sutil desses achados.

## 5. Práticas de mitigação propostas

1. **Balanceamento por subgrupo** — equilibrar bacteriana/viral no treino, não só
   NORMAL/PNEUMONIA.
2. **Reponderação (`class_weight`) ou *oversampling*** do subgrupo minoritário (viral).
3. **Ajuste de limiar por sensibilidade clínica** — em triagem, reduzir o limiar de
   0,5 eleva o recall (menos falsos negativos), aceitando mais falsos positivos; a
   escolha do ponto de operação é uma decisão de governança clínica.
4. **Coleta mais representativa** — adultos/idosos, múltiplos centros e equipamentos.
5. **Governança contínua (MLOps)** — monitorar métricas **por subgrupo** em produção
   (detecção de *drift* e de degradação por grupo), registrar predições de forma
   **auditável**, manter **humano no circuito** e tratar os dados conforme a **LGPD**.

## 6. Conclusão

O modelo é um protótipo de **apoio à decisão** com vieses conhecidos e documentados:
representatividade pediátrica/regional e menor sensibilidade à pneumonia viral.
Tornar esses limites explícitos — e medi-los com a métrica de fairness correta — é
parte essencial de uma IA responsável em saúde. As mitigações acima são o caminho
para reduzir o risco antes de qualquer uso real.

*Reprodutível em `notebooks/ir_alem1_fairness.ipynb`; figura em `docs/ir_alem1_fairness.png`; métricas em `src/fairness.py` (com testes em `tests/test_fairness.py`).*
