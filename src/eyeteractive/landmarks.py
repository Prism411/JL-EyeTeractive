"""Extração de landmarks oculares via MediaPipe FaceLandmarker (API Tasks).

O protótipo original usava dois detectores de face por frame: ``dlib`` para
recortar o olho que alimentava a CNN e ``mediapipe.solutions.face_mesh`` para
os landmarks do contorno. Além do custo duplicado, as duas fontes enxergavam
recortes ligeiramente diferentes do mesmo olho, o que introduzia desacordo
espúrio entre elas.

Aqui há um único detector. Do mesmo conjunto de landmarks saem o contorno para
a geometria, o contorno da íris e o recorte que vai para a rede — de modo que
qualquer contradição observada entre as fontes é informação real sobre o
olhar, não artefato de alinhamento.

A API ``mediapipe.solutions`` foi removida no MediaPipe 1.0; este módulo usa
``mediapipe.tasks.python.vision.FaceLandmarker``, que devolve os mesmos 478
pontos da malha canônica (468 do rosto + 10 da íris).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = [
    "OLHO_DIREITO",
    "OLHO_ESQUERDO",
    "IRIS_DIREITA",
    "IRIS_ESQUERDA",
    "DeteccaoOcular",
    "DetectorFacial",
]

# Contornos na ordem exigida por :mod:`eyeteractive.geometry`:
# canto externo, superior×2, canto interno, inferior×2.
#
# Atenção à nomenclatura: estes são os olhos da *pessoa*. O olho direito dela
# aparece do lado esquerdo da imagem. O protótipo chamava este mesmo conjunto
# de ``left_eye_indices``, embora fosse o olho que o dlib tratava como
# ``right_eye`` — a CNN e a geometria sempre olharam o mesmo olho, apenas com
# nomes trocados.
OLHO_DIREITO: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
OLHO_ESQUERDO: tuple[int, ...] = (263, 387, 385, 362, 380, 373)

IRIS_DIREITA: tuple[int, ...] = (469, 470, 471, 472)
IRIS_ESQUERDA: tuple[int, ...] = (474, 475, 476, 477)

# Margem, em pixels, ao redor da caixa do olho no recorte enviado à rede.
# Reproduz a margem usada na extração do dataset de treino.
MARGEM_RECORTE = 20


@dataclass(frozen=True)
class DeteccaoOcular:
    """Landmarks e recorte de um olho num frame."""

    contorno: tuple[tuple[float, float], ...]
    iris: tuple[tuple[float, float], ...]
    recorte_rgb: np.ndarray
    caixa: tuple[int, int, int, int]  # x, y, largura, altura


class DetectorFacial:
    """Envoltório do FaceLandmarker configurado para um único rosto."""

    def __init__(
        self,
        modelo: str | Path,
        modo_video: bool = True,
        confianca_minima: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        caminho = Path(modelo)
        if not caminho.is_file():
            raise FileNotFoundError(
                f"Modelo do FaceLandmarker não encontrado: {caminho}. "
                "Baixe-o com scripts/baixar_modelos.sh"
            )

        self._mp = mp
        self._modo_video = modo_video
        self._ultimo_timestamp = -1
        self._criar = None  # definido abaixo, para permitir recriação
        opcoes = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(caminho)),
            running_mode=RunningMode.VIDEO if modo_video else RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=confianca_minima,
            min_face_presence_confidence=confianca_minima,
            min_tracking_confidence=confianca_minima,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._criar = lambda: FaceLandmarker.create_from_options(opcoes)
        self._detector = self._criar()

    def __enter__(self) -> "DetectorFacial":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()

    def fechar(self) -> None:
        self._detector.close()

    def reiniciar(self) -> None:
        """Recria o detector, zerando rastreamento e relógio.

        Necessário ao trocar de vídeo: no modo VIDEO o FaceLandmarker exige
        timestamps estritamente crescentes durante toda a vida da instância, e
        o rastreamento carregado de um vídeo não faz sentido no seguinte.
        """
        self._detector.close()
        self._detector = self._criar()
        self._ultimo_timestamp = -1

    def detectar(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int = 0,
        olho: Sequence[int] = OLHO_DIREITO,
        iris: Sequence[int] = IRIS_DIREITA,
    ) -> DeteccaoOcular | None:
        """Detecta o olho indicado; devolve ``None`` se nenhum rosto for achado."""
        import cv2

        mp = self._mp
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self._modo_video:
            # Frames repetidos ou fora de ordem (vídeo de taxa variável,
            # amostragem por passo) derrubariam o detector com
            # "Input timestamp must be monotonically increasing".
            if timestamp_ms <= self._ultimo_timestamp:
                timestamp_ms = self._ultimo_timestamp + 1
            self._ultimo_timestamp = timestamp_ms
            resultado = self._detector.detect_for_video(imagem, timestamp_ms)
        else:
            resultado = self._detector.detect(imagem)

        if not resultado.face_landmarks:
            return None

        pontos = resultado.face_landmarks[0]
        altura, largura = frame_bgr.shape[:2]

        def escalar(indices: Sequence[int]) -> tuple[tuple[float, float], ...]:
            return tuple(
                (pontos[i].x * largura, pontos[i].y * altura) for i in indices
            )

        contorno = escalar(olho)
        contorno_iris = escalar(iris)

        recorte, caixa = self._recortar(frame_rgb, contorno)
        if recorte is None:
            return None

        return DeteccaoOcular(
            contorno=contorno,
            iris=contorno_iris,
            recorte_rgb=recorte,
            caixa=caixa,
        )

    @staticmethod
    def _recortar(
        frame_rgb: np.ndarray, contorno: Sequence[tuple[float, float]]
    ) -> tuple[np.ndarray | None, tuple[int, int, int, int]]:
        """Recorta a caixa do olho com margem, respeitando as bordas do frame."""
        altura, largura = frame_rgb.shape[:2]
        xs = [p[0] for p in contorno]
        ys = [p[1] for p in contorno]

        x0 = max(0, int(min(xs)) - MARGEM_RECORTE)
        y0 = max(0, int(min(ys)) - MARGEM_RECORTE)
        x1 = min(largura, int(max(xs)) + MARGEM_RECORTE)
        y1 = min(altura, int(max(ys)) + MARGEM_RECORTE)

        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, (x0, y0, 0, 0)

        return frame_rgb[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)
