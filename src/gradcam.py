"""
Projeto : CardioIA — Fase 4 / Cap 1 (Visão Computacional na Clínica)
Módulo  : Grad-CAM — interpretabilidade da classificação
Autor   : Carlos Mário Vieira de Melo Filho — RM563769
Data    : 2026-06-16

Gera um mapa de calor (Grad-CAM) destacando as regiões da imagem que mais
influenciaram a predição da CNN. É o que dá o caráter de "informação
interpretável" pedido no enunciado: o clínico vê ONDE o modelo olhou.

Requer que os modelos sejam grafos planos (no projeto, a base VGG16 é ligada via
`input_tensor`, então `model.get_layer("block5_conv3")` é acessível diretamente).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

matplotlib.use("Agg")  # backend sem display (Flask / notebook headless)


def _ultima_camada_conv(model: tf.keras.Model) -> str:
    """Nome da última camada convolucional do modelo.

    Prioriza a última ``Conv2D``; se não houver, cai para a última camada com
    saída espacial 4D (ex.: MaxPooling2D). Robusto no Keras 3 (usa output.shape).
    """
    fallback: str | None = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if fallback is None:
            try:
                if len(layer.output.shape) == 4:
                    fallback = layer.name
            except (AttributeError, ValueError):
                pass
    if fallback:
        return fallback
    raise ValueError("Nenhuma camada convolucional encontrada no modelo.")


def gerar_heatmap(
    batch: np.ndarray,
    model: tf.keras.Model,
    nome_camada_conv: str | None = None,
) -> np.ndarray:
    """Calcula o heatmap Grad-CAM para um batch ``(1, H, W, 3)``.

    Retorna um array 2D normalizado em [0, 1] no tamanho do mapa de ativação.
    """
    nome = nome_camada_conv or _ultima_camada_conv(model)
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(nome).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(batch)
        # Saída binária (sigmoide, 1 neurônio): o escore é a própria predição.
        score = preds[:, 0]

    grads = tape.gradient(score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # importância por canal

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def sobrepor(
    img_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Sobrepõe o heatmap (colormap 'jet') sobre a imagem original.

    ``img_array``: RGB float32 [0, 255] no tamanho de entrada do modelo.
    Retorna uma imagem PIL pronta para exibir/salvar.
    """
    h, w = img_array.shape[:2]

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (w, h), Image.BILINEAR
    )
    heatmap_resized = np.asarray(heatmap_img) / 255.0

    colormap = matplotlib.colormaps["jet"]
    colored = colormap(heatmap_resized)[..., :3]  # descarta canal alfa → RGB

    base = img_array / 255.0
    overlay = (1 - alpha) * base + alpha * colored
    overlay = np.uint8(255 * np.clip(overlay, 0, 1))
    return Image.fromarray(overlay)
