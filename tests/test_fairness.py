"""Testes das métricas de fairness (Ir Além 1) — cálculo puro, sem TF."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import fairness


def test_subtipo_pneumonia():
    assert fairness.subtipo_pneumonia("person104_bacteria_491.jpeg") == "bacteriana"
    assert fairness.subtipo_pneumonia("person1097_virus_1817.jpeg") == "viral"
    assert fairness.subtipo_pneumonia("IM-0007-0001.jpeg") == "normal"


def test_metricas_por_grupo_recall_e_fnr():
    # Grupo 'viral': 2 positivos, 1 detectado -> recall 0.5, fnr 0.5
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0])
    grupos = np.array(["bacteriana", "bacteriana", "viral", "viral"])

    m = fairness.metricas_por_grupo(y_true, y_pred, grupos)
    assert m["bacteriana"]["recall"] == 1.0
    assert m["viral"]["recall"] == 0.5
    assert m["viral"]["fnr"] == 0.5
    assert m["bacteriana"]["n"] == 2


def test_diferenca_de_oportunidade():
    eod = fairness.diferenca_de_oportunidade({"bacteriana": 1.0, "viral": 0.5})
    assert eod == 0.5
    # Um único grupo -> sem disparidade.
    assert fairness.diferenca_de_oportunidade({"bacteriana": 1.0}) == 0.0


def test_impacto_desigual():
    di = fairness.impacto_desigual({"a": 0.4, "b": 0.8})
    assert abs(di - 0.5) < 1e-9
