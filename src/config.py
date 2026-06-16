"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Módulo  : Configuração central (fonte única de verdade de caminhos e hiperparâmetros)
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Mantém em um só lugar as constantes usadas pelos notebooks E pelo app Flask,
evitando divergência de tamanho de imagem / classes entre treino e inferência.
"""

from __future__ import annotations

from pathlib import Path

# --- Caminhos do projeto (relativos à raiz do repositório) -------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"

# Dataset completo (Kermany) e o subset reduzido usado no treino local em CPU.
DATASET_DIR = DATA_DIR / "chest_xray"
SUBSET_DIR = DATA_DIR / "chest_xray_subset"

# --- Parâmetros de imagem ----------------------------------------------------
# 224x224x3 atende tanto à CNN do zero quanto ao VGG16 (entrada esperada pelo
# pré-treino ImageNet), permitindo um único pipeline de pré-processamento.
IMG_SIZE: tuple[int, int] = (224, 224)
IMG_CHANNELS = 3

# --- Classes (ordem fixa = índice do rótulo) ---------------------------------
# Alfabética, igual ao que o image_dataset_from_directory infere das pastas.
CLASSES: list[str] = ["NORMAL", "PNEUMONIA"]

# --- Treino ------------------------------------------------------------------
BATCH_SIZE = 32
SEED = 42

# --- Artefatos de modelo -----------------------------------------------------
MODEL_SCRATCH = MODELS_DIR / "cnn_scratch.keras"
MODEL_TRANSFER = MODELS_DIR / "cnn_transfer_vgg16.keras"

# Rótulos amigáveis para a interface (apoio à decisão, não diagnóstico).
CLASSE_LABEL_PT = {
    "NORMAL": "Sem sinais de pneumonia",
    "PNEUMONIA": "Padrão sugestivo de pneumonia",
}
