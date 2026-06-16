"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Módulo  : Pré-processamento de imagens (Parte 1) — fonte única de verdade
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Centraliza TODO o pré-processamento para que treino (notebooks) e inferência
(app Flask) usem exatamente a mesma transformação, evitando train/serve skew.

Decisão de projeto: a normalização específica de cada modelo (Rescaling 1/255
para a CNN do zero; vgg16.preprocess_input para o Transfer Learning) é embutida
COMO PRIMEIRA CAMADA do respectivo modelo, no notebook da Parte 2. Aqui apenas
carregamos, redimensionamos e convertemos para RGB em float32 [0, 255]; assim um
único array serve a qualquer um dos modelos.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np
import tensorflow as tf
from PIL import Image

from . import config


def make_datasets(
    subset_dir: Path = config.SUBSET_DIR,
    img_size: tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Cria os datasets de treino, validação e teste a partir das pastas.

    Espera a estrutura ``subset_dir/{train,val,test}/{NORMAL,PNEUMONIA}``.
    As imagens saem em RGB float32 [0, 255]; a normalização fica no modelo.
    """
    if not subset_dir.exists():
        raise FileNotFoundError(
            f"Subset não encontrado em {subset_dir}. "
            "Rode `make dataset` (scripts/download_dataset.py) antes."
        )

    def _load(split: str, shuffle: bool) -> tf.data.Dataset:
        return tf.keras.utils.image_dataset_from_directory(
            subset_dir / split,
            labels="inferred",
            label_mode="binary",          # 1 neurônio sigmoide na saída
            class_names=config.CLASSES,   # garante NORMAL=0, PNEUMONIA=1
            color_mode="rgb",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = _load("train", shuffle=True)
    val_ds = _load("val", shuffle=False)
    test_ds = _load("test", shuffle=False)

    # Prefetch para sobrepor I/O e computação durante o treino.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds


def build_augmentation() -> tf.keras.Sequential:
    """Camadas de data augmentation (ativas só no treino).

    Aumentos conservadores: radiografias têm orientação clínica, então evitamos
    flip vertical; usamos flip horizontal leve, pequena rotação e zoom.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=config.SEED),
            tf.keras.layers.RandomRotation(0.05, seed=config.SEED),
            tf.keras.layers.RandomZoom(0.10, seed=config.SEED),
        ],
        name="data_augmentation",
    )


def load_image_array(
    source: str | Path | BinaryIO | Image.Image,
    img_size: tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """Carrega uma imagem (caminho, file-like do upload ou PIL) → array RGB.

    Retorna float32 com shape ``(H, W, 3)`` em [0, 255], pronto para qualquer
    um dos modelos (a normalização é feita dentro do modelo).
    """
    if isinstance(source, Image.Image):
        img = source
    else:
        img = Image.open(source)
    img = img.convert("RGB").resize(img_size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def preprocess_single(
    source: str | Path | BinaryIO | Image.Image,
    img_size: tuple[int, int] = config.IMG_SIZE,
) -> np.ndarray:
    """Igual a :func:`load_image_array`, mas com dimensão de batch ``(1, H, W, 3)``."""
    arr = load_image_array(source, img_size=img_size)
    return np.expand_dims(arr, axis=0)
