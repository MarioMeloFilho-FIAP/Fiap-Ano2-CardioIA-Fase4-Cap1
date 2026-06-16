"""Testes do pré-processamento (Parte 1) — shape, canais e faixa de valores."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, preprocessing


def _imagem_falsa_grayscale(tmp_path: Path) -> Path:
    """Cria um PNG em tons de cinza (simula radiografia de 1 canal)."""
    arr = (np.random.rand(64, 80) * 255).astype("uint8")
    caminho = tmp_path / "fake_xray.png"
    Image.fromarray(arr, mode="L").save(caminho)
    return caminho


def test_load_image_array_shape_e_canais(tmp_path):
    caminho = _imagem_falsa_grayscale(tmp_path)
    arr = preprocessing.load_image_array(caminho)

    assert arr.shape == (*config.IMG_SIZE, 3)  # redimensionada e convertida p/ RGB
    assert arr.dtype == np.float32


def test_load_image_array_faixa_de_valores(tmp_path):
    caminho = _imagem_falsa_grayscale(tmp_path)
    arr = preprocessing.load_image_array(caminho)

    # A normalização fica no modelo; aqui os pixels seguem em [0, 255].
    assert arr.min() >= 0.0
    assert arr.max() <= 255.0


def test_preprocess_single_adiciona_batch(tmp_path):
    caminho = _imagem_falsa_grayscale(tmp_path)
    batch = preprocessing.preprocess_single(caminho)

    assert batch.shape == (1, *config.IMG_SIZE, 3)
