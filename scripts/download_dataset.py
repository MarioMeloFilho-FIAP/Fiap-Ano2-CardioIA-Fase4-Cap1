"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Script  : Download do dataset Chest X-ray (Kermany) + geração de subset reduzido
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Baixa o dataset público de raios-X de tórax (Normal vs. Pneumonia) e monta um
SUBSET BALANCEADO e reduzido, adequado para treino local em CPU.

Uso:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --train 400 --val 100 --test 100

Origem dos dados:
    Kaggle: paultimothymooney/chest-xray-pneumonia
    (Kermany et al., 2018 — "Labeled OCT and Chest X-Ray Images for Classification")

Sem acesso ao Kaggle? Baixe/descompacte manualmente e aponte a pasta que contém
'chest_xray/' via variável de ambiente:
    export CHEST_XRAY_DIR=/caminho/para/chest_xray
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

# Permite rodar o script direto (python scripts/...) achando o pacote src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
EXTS = {".jpeg", ".jpg", ".png"}


def localizar_dataset_bruto() -> Path:
    """Retorna a pasta 'chest_xray' (com train/val/test), baixando se preciso."""
    # 1) Override manual.
    env_dir = os.environ.get("CHEST_XRAY_DIR")
    if env_dir:
        base = Path(env_dir)
        return base if base.name == "chest_xray" else _achar_chest_xray(base)

    # 2) Download via kagglehub (cacheado entre execuções).
    try:
        import kagglehub
    except ImportError:
        sys.exit(
            "kagglehub não instalado. Rode `make prep-venv` ou defina CHEST_XRAY_DIR."
        )

    print(f">> Baixando '{KAGGLE_DATASET}' via kagglehub (pode demorar na 1ª vez)...")
    caminho = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    return _achar_chest_xray(caminho)


def _achar_chest_xray(base: Path) -> Path:
    """Procura recursivamente uma pasta 'chest_xray' que contenha 'train'."""
    if (base / "train").exists():
        return base
    for p in base.rglob("chest_xray"):
        if (p / "train").exists():
            return p
    sys.exit(f"Não encontrei a estrutura 'chest_xray/train' em {base}.")


def _imagens_por_classe(raw_dir: Path, classe: str) -> list[Path]:
    """Junta as imagens de uma classe vindas de train/val/test do dataset bruto."""
    arquivos: list[Path] = []
    for split in ("train", "val", "test"):
        pasta = raw_dir / split / classe
        if pasta.exists():
            arquivos += [p for p in pasta.iterdir() if p.suffix.lower() in EXTS]
    return arquivos


def gerar_subset(raw_dir: Path, n_train: int, n_val: int, n_test: int) -> None:
    """Cria data/chest_xray_subset/{train,val,test}/{NORMAL,PNEUMONIA} balanceado."""
    destino = config.SUBSET_DIR
    rng = random.Random(config.SEED)
    splits = {"train": n_train, "val": n_val, "test": n_test}

    if destino.exists():
        shutil.rmtree(destino)

    for classe in config.CLASSES:
        disponiveis = _imagens_por_classe(raw_dir, classe)
        rng.shuffle(disponiveis)
        total_pedido = sum(splits.values())
        if len(disponiveis) < total_pedido:
            print(
                f"   [aviso] classe {classe}: só {len(disponiveis)} imagens "
                f"(pedido {total_pedido}); usando todas as disponíveis."
            )

        idx = 0
        for split, n in splits.items():
            pasta_out = destino / split / classe
            pasta_out.mkdir(parents=True, exist_ok=True)
            selecao = disponiveis[idx : idx + n]
            idx += n
            for origem in selecao:
                shutil.copy2(origem, pasta_out / origem.name)
            print(f"   {split}/{classe}: {len(selecao)} imagens")

    print(f">> Subset gerado em {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara o subset de raios-X.")
    parser.add_argument("--train", type=int, default=400, help="imagens/classe (treino)")
    parser.add_argument("--val", type=int, default=100, help="imagens/classe (validação)")
    parser.add_argument("--test", type=int, default=100, help="imagens/classe (teste)")
    args = parser.parse_args()

    raw_dir = localizar_dataset_bruto()
    print(f">> Dataset bruto em: {raw_dir}")
    gerar_subset(raw_dir, args.train, args.val, args.test)
    print(">> Pronto. Agora rode os notebooks (Parte 1 e Parte 2) ou `make train`.")


if __name__ == "__main__":
    main()
