"""Smoke test da inferência (Parte 2) com um modelo dummy treinável-rápido.

Não depende dos modelos treinados nem do dataset: monta uma CNN mínima com o
mesmo contrato de entrada/saída e verifica o fluxo classificar() + Grad-CAM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, gradcam, inference


def _modelo_dummy() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, config.IMG_CHANNELS))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def test_classificar_salva_e_carrega(tmp_path, monkeypatch):
    modelo = _modelo_dummy()
    caminho = tmp_path / "dummy.keras"
    modelo.save(caminho)

    # Aponta o registro de modelos para o dummy e limpa o cache de carga.
    monkeypatch.setitem(inference.MODELOS, "dummy", caminho)
    inference.carregar_modelo.cache_clear()

    batch = np.random.rand(1, *config.IMG_SIZE, 3).astype("float32") * 255
    pred = inference.classificar(batch, nome_modelo="dummy")

    assert pred.classe in config.CLASSES
    assert 0.5 <= pred.probabilidade <= 1.0  # confiança da classe vencedora
    assert 0.0 <= pred.prob_pneumonia <= 1.0


def test_gradcam_gera_heatmap_normalizado():
    modelo = _modelo_dummy()
    batch = np.random.rand(1, *config.IMG_SIZE, 3).astype("float32") * 255

    heatmap = gradcam.gerar_heatmap(batch, modelo)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0

    overlay = gradcam.sobrepor(batch[0], heatmap)
    assert overlay.size == config.IMG_SIZE  # (largura, altura)
