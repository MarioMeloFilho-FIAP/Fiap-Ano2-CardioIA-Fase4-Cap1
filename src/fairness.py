"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Módulo  : Métricas de fairness por subgrupo (Ir Além 1 — Ética e Governança)
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Mede se o classificador trata subgrupos de forma equitativa. No dataset Kermany,
a classe PNEUMONIA traz o subtipo no nome do arquivo (bacteriana/viral); usamos
esse eixo como subgrupo sensível para uma análise de *equal opportunity*: o modelo
detecta pneumonia viral tão bem quanto a bacteriana?

As métricas são implementadas "na mão" (numpy) — sem dependências pesadas de
fairness — para transparência e auditabilidade do cálculo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def subtipo_pneumonia(nome_arquivo: str | Path) -> str:
    """Extrai o subgrupo a partir do nome do arquivo Kermany.

    Ex.: 'person104_bacteria_491.jpeg' -> 'bacteriana';
         'person1097_virus_1817.jpeg'  -> 'viral';
         'IM-0007-0001.jpeg'           -> 'normal'.
    """
    nome = Path(nome_arquivo).name.lower()
    if "bacteria" in nome:
        return "bacteriana"
    if "virus" in nome:
        return "viral"
    return "normal"


def metricas_por_grupo(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grupos: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Calcula métricas por subgrupo.

    Parâmetros são arrays alinhados (mesmo índice = mesmo exemplo). ``y_true`` e
    ``y_pred`` são binários (1 = PNEUMONIA). ``grupos`` rotula o subgrupo de cada
    exemplo. Retorna, por grupo: n, acurácia, taxa de positivos previstos
    (selection rate) e — quando há positivos reais no grupo — recall (TPR) e
    taxa de falsos negativos (FNR).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    grupos = np.asarray(grupos)

    resultado: dict[str, dict[str, float]] = {}
    for g in sorted(set(grupos)):
        mask = grupos == g
        yt, yp = y_true[mask], y_pred[mask]
        n = int(mask.sum())
        info: dict[str, float] = {
            "n": n,
            "acuracia": float((yt == yp).mean()) if n else float("nan"),
            "taxa_positivos_previstos": float((yp == 1).mean()) if n else float("nan"),
        }
        positivos = int((yt == 1).sum())
        if positivos:
            tpr = float(((yp == 1) & (yt == 1)).sum() / positivos)  # recall
            info["recall"] = tpr
            info["fnr"] = 1.0 - tpr  # falsos negativos: o que é mais grave em saúde
        resultado[str(g)] = info
    return resultado


def diferenca_de_oportunidade(recalls: dict[str, float]) -> float:
    """Equal Opportunity Difference: maior diferença de recall entre subgrupos.

    0 = paridade perfeita de recall; quanto maior, mais o modelo "perde" mais
    casos de um subgrupo que de outro.
    """
    valores = [v for v in recalls.values() if not np.isnan(v)]
    if len(valores) < 2:
        return 0.0
    return float(max(valores) - min(valores))


def impacto_desigual(taxas_selecao: dict[str, float]) -> float:
    """Disparate Impact: razão entre a menor e a maior taxa de seleção.

    1,0 = paridade; valores abaixo de ~0,8 são o limiar clássico de alerta
    (regra dos 80%).
    """
    valores = [v for v in taxas_selecao.values() if not np.isnan(v) and v > 0]
    if len(valores) < 2:
        return 1.0
    return float(min(valores) / max(valores))
