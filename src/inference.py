"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Módulo  : Inferência — carrega um modelo .keras e classifica uma imagem
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Usado pelo app Flask. Mantém um cache simples de modelos para não recarregar o
.keras a cada requisição.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from . import config

# Modelos disponíveis para o usuário escolher na interface.
MODELOS: dict[str, Path] = {
    "scratch": config.MODEL_SCRATCH,
    "transfer": config.MODEL_TRANSFER,
}


@dataclass
class Predicao:
    """Resultado de uma classificação para apresentação na interface."""

    classe: str            # "NORMAL" | "PNEUMONIA"
    rotulo: str            # texto amigável em PT-BR
    probabilidade: float   # confiança da classe prevista, em [0, 1]
    prob_pneumonia: float  # saída bruta do sigmoide (P(PNEUMONIA))


@lru_cache(maxsize=4)
def carregar_modelo(nome: str) -> tf.keras.Model:
    """Carrega (e memoiza) um modelo treinado pelo nome lógico."""
    if nome not in MODELOS:
        raise ValueError(f"Modelo '{nome}' inválido. Opções: {list(MODELOS)}")
    caminho = MODELOS[nome]
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. "
            "Treine antes via `make train` (notebooks/parte2)."
        )
    return tf.keras.models.load_model(caminho)


def classificar(batch: np.ndarray, nome_modelo: str = "transfer") -> Predicao:
    """Classifica um batch já pré-processado ``(1, H, W, 3)`` em [0, 255].

    O modelo embute sua própria normalização (Rescaling ou vgg16.preprocess_input),
    então recebemos o array bruto vindo de ``preprocessing.preprocess_single``.
    """
    modelo = carregar_modelo(nome_modelo)
    prob_pneumonia = float(modelo.predict(batch, verbose=0).ravel()[0])

    if prob_pneumonia >= 0.5:
        classe = "PNEUMONIA"
        confianca = prob_pneumonia
    else:
        classe = "NORMAL"
        confianca = 1.0 - prob_pneumonia

    return Predicao(
        classe=classe,
        rotulo=config.CLASSE_LABEL_PT[classe],
        probabilidade=confianca,
        prob_pneumonia=prob_pneumonia,
    )
