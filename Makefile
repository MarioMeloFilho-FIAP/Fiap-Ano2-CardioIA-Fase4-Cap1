SHELL := /bin/bash

# venv dedicado em Python 3.11 (TensorFlow não suporta o python3 padrão = 3.14)
VENV_NAME := fiap_ano2_fase4_cap1_venv
VENV_BIN  := $(VENV_NAME)/bin
PYTHON    := $(VENV_BIN)/python3
PIP       := $(VENV_BIN)/pip
PY311     := python3.11

.DEFAULT_GOAL := help

.PHONY: all prep-venv shell dataset jupyter train web test clean help

all: prep-venv
	source $(VENV_BIN)/activate && /bin/bash

prep-venv:
	$(PY311) -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt --quiet
	@echo ">> venv '$(VENV_NAME)' pronto (Python 3.11 + TensorFlow)."

dataset:
	source $(VENV_BIN)/activate && $(PYTHON) scripts/download_dataset.py

jupyter:
	source $(VENV_BIN)/activate && jupyter notebook notebooks/

# Executa os notebooks de ponta a ponta (gera os modelos .keras e os PNGs de métricas)
train:
	source $(VENV_BIN)/activate && \
		jupyter nbconvert --to notebook --execute --inplace notebooks/parte1_preprocessamento.ipynb && \
		jupyter nbconvert --to notebook --execute --inplace notebooks/parte2_cnn_classificacao.ipynb

web:
	source $(VENV_BIN)/activate && $(PYTHON) -m webapp.app

test:
	source $(VENV_BIN)/activate && $(PYTHON) -m pytest tests -q

clean:
	-rm -rf $(VENV_NAME)

help:
	@echo "Targets disponíveis (execute com 'make <target>'):"
	@echo "  prep-venv   Cria o venv '$(VENV_NAME)' (Python 3.11) e instala requirements"
	@echo "  dataset     Baixa o Chest X-ray (Kermany) e gera o subset reduzido em data/"
	@echo "  jupyter     Abre o Jupyter Notebook apontando para notebooks/"
	@echo "  train       Executa parte1 e parte2 (gera models/*.keras e métricas em docs/)"
	@echo "  web         Sobe o app Flask (http://localhost:5000)"
	@echo "  test        Roda a suíte de testes (tests/) dentro do venv"
	@echo "  clean       Remove o venv '$(VENV_NAME)'"
	@echo "  all         prep-venv + abre um shell com o venv ativado"
	@echo "  help        Exibe esta mensagem de ajuda"
