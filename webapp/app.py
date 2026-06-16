"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Módulo  : App Flask — protótipo de triagem visual de raios-X de tórax
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Interface web simples: o usuário envia uma radiografia, escolhe o modelo
(CNN do zero ou Transfer Learning VGG16) e recebe a classificação, a confiança
e um mapa de calor Grad-CAM indicando as regiões que pesaram na decisão.

Reaproveita o MESMO pré-processamento dos notebooks (src.preprocessing),
garantindo coerência entre treino e inferência.

Aviso clínico: ferramenta acadêmica de APOIO à decisão — não substitui laudo
médico nem constitui diagnóstico.
"""

from __future__ import annotations

import base64
import io

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from src import config, gradcam, inference, preprocessing


def _img_para_data_uri(img: Image.Image) -> str:
    """Converte uma imagem PIL em data URI base64 para embutir no HTML."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB por upload

    @app.route("/")
    def index():
        # Quais modelos já foram treinados (existe .keras)?
        disponiveis = {
            nome: caminho.exists() for nome, caminho in inference.MODELOS.items()
        }
        return render_template("index.html", disponiveis=disponiveis)

    @app.route("/predict", methods=["POST"])
    def predict():
        arquivo = request.files.get("imagem")
        modelo = request.form.get("modelo", "transfer")

        if arquivo is None or arquivo.filename == "":
            return render_template("index.html", erro="Selecione uma imagem.",
                                   disponiveis=_status_modelos()), 400

        try:
            img_pil = Image.open(arquivo.stream)
            img_pil.load()
        except UnidentifiedImageError:
            return render_template("index.html", erro="Arquivo de imagem inválido.",
                                   disponiveis=_status_modelos()), 400

        try:
            batch = preprocessing.preprocess_single(img_pil)
            pred = inference.classificar(batch, nome_modelo=modelo)
            modelo_keras = inference.carregar_modelo(modelo)
            heatmap = gradcam.gerar_heatmap(batch, modelo_keras)
            overlay = gradcam.sobrepor(batch[0], heatmap)
        except FileNotFoundError as exc:
            return render_template("index.html", erro=str(exc),
                                   disponiveis=_status_modelos()), 503

        # Imagem original redimensionada (o que o modelo realmente "vê").
        original = Image.fromarray(batch[0].astype("uint8"))

        return render_template(
            "resultado.html",
            pred=pred,
            modelo=modelo,
            img_original=_img_para_data_uri(original),
            img_gradcam=_img_para_data_uri(overlay),
            classes=config.CLASSES,
        )

    return app


def _status_modelos() -> dict[str, bool]:
    return {nome: caminho.exists() for nome, caminho in inference.MODELOS.items()}


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
