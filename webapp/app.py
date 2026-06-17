"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Módulo  : App Flask — protótipo web + API JSON (para o app mobile, Ir Além 2)
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Expõe duas faces sobre a MESMA lógica de inferência:
- Interface web (Jinja): rotas `/` e `/predict`.
- API JSON (consumida pelo app React Native): `GET /api/health`,
  `GET /api/models` e `POST /api/predict`.

Reaproveita o mesmo pré-processamento dos notebooks (src.preprocessing),
garantindo coerência entre treino e inferência.

Aviso clínico: ferramenta acadêmica de APOIO à decisão — não substitui laudo
médico nem constitui diagnóstico.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

from src import config, gradcam, inference, preprocessing


@dataclass
class ResultadoAnalise:
    """Agrega tudo que uma predição produz, para web e API reaproveitarem."""

    pred: inference.Predicao
    modelo: str
    original_uri: str
    gradcam_uri: str


def _img_para_data_uri(img: Image.Image) -> str:
    """Converte uma imagem PIL em data URI base64 (embutível em HTML ou JSON)."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _status_modelos() -> dict[str, bool]:
    return {nome: caminho.exists() for nome, caminho in inference.MODELOS.items()}


def _analisar(arquivo, modelo: str) -> ResultadoAnalise:
    """Núcleo da predição, compartilhado entre a web e a API.

    Lança ``UnidentifiedImageError`` (imagem inválida) ou ``FileNotFoundError``
    (modelo não treinado) — tratados por cada rota.
    """
    img_pil = Image.open(arquivo)
    img_pil.load()

    batch = preprocessing.preprocess_single(img_pil)
    pred = inference.classificar(batch, nome_modelo=modelo)
    modelo_keras = inference.carregar_modelo(modelo)
    heatmap = gradcam.gerar_heatmap(batch, modelo_keras)
    overlay = gradcam.sobrepor(batch[0], heatmap)
    original = Image.fromarray(batch[0].astype("uint8"))

    return ResultadoAnalise(
        pred=pred,
        modelo=modelo,
        original_uri=_img_para_data_uri(original),
        gradcam_uri=_img_para_data_uri(overlay),
    )


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB por upload
    CORS(app, resources={r"/api/*": {"origins": "*"}})  # libera a API p/ o mobile

    # ---------------------------- Interface web ----------------------------
    @app.route("/")
    def index():
        return render_template("index.html", disponiveis=_status_modelos())

    @app.route("/predict", methods=["POST"])
    def predict():
        arquivo = request.files.get("imagem")
        modelo = request.form.get("modelo", "transfer")

        if arquivo is None or arquivo.filename == "":
            return render_template("index.html", erro="Selecione uma imagem.",
                                   disponiveis=_status_modelos()), 400
        try:
            res = _analisar(arquivo.stream, modelo)
        except UnidentifiedImageError:
            return render_template("index.html", erro="Arquivo de imagem inválido.",
                                   disponiveis=_status_modelos()), 400
        except FileNotFoundError as exc:
            return render_template("index.html", erro=str(exc),
                                   disponiveis=_status_modelos()), 503

        return render_template(
            "resultado.html",
            pred=res.pred,
            modelo=res.modelo,
            img_original=res.original_uri,
            img_gradcam=res.gradcam_uri,
            classes=config.CLASSES,
        )

    # ------------------------------- API JSON -------------------------------
    @app.get("/api/health")
    def api_health():
        """Checagem de saúde + quais modelos estão disponíveis."""
        return jsonify(status="ok", modelos=_status_modelos(), classes=config.CLASSES)

    @app.get("/api/models")
    def api_models():
        return jsonify(modelos=_status_modelos())

    @app.post("/api/predict")
    def api_predict():
        """Recebe uma imagem (multipart, campo 'imagem') e devolve a classificação."""
        arquivo = request.files.get("imagem")
        modelo = request.form.get("modelo", "transfer")
        incluir_gradcam = request.args.get("gradcam", "1") != "0"

        if arquivo is None or arquivo.filename == "":
            return jsonify(erro="Envie uma imagem no campo 'imagem'."), 400
        try:
            res = _analisar(arquivo.stream, modelo)
        except UnidentifiedImageError:
            return jsonify(erro="Arquivo de imagem inválido."), 400
        except ValueError as exc:           # modelo inexistente
            return jsonify(erro=str(exc)), 400
        except FileNotFoundError as exc:    # modelo não treinado
            return jsonify(erro=str(exc)), 503

        payload = {
            "classe": res.pred.classe,
            "rotulo": res.pred.rotulo,
            "probabilidade": round(res.pred.probabilidade, 4),
            "prob_pneumonia": round(res.pred.prob_pneumonia, 4),
            "modelo": res.modelo,
            "classes": config.CLASSES,
        }
        if incluir_gradcam:
            payload["gradcam"] = res.gradcam_uri
        return jsonify(payload)

    return app


if __name__ == "__main__":
    # host=0.0.0.0 para que o celular na mesma rede alcance a API.
    create_app().run(host="0.0.0.0", port=5000, debug=True)
