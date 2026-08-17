"""Pipeline de ponta a ponta: frame de vídeo → direção do olhar.

Encadeia detecção de landmarks, geometria, CNN e fusão paraconsistente atrás
de uma única chamada, de modo que servidor, webcam e avaliação offline
compartilhem exatamente o mesmo caminho de decisão.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .cnn import ClassificadorOlhar, MarginaisCNN
from .fusion import ConfigFusao, Direcao, FusorParaconsistente, ResultadoOlhar
from .geometry import MedidasOculares, medir
from .landmarks import IRIS_DIREITA, OLHO_DIREITO, DetectorFacial

__all__ = ["QuadroAnalisado", "PipelineOlhar", "DIRECAO_PARA_CODIGO", "codigo_de_direcao"]

CAMINHO_PADRAO_LANDMARKER = Path("models/face_landmarker.task")
CAMINHO_PADRAO_PESOS = Path("models/resnet101_model.pth")

# Códigos do protocolo legado do servidor, preservados para não quebrar o
# aplicativo móvel já publicado: 0 = sem comando, 1..4 = cardeais.
DIRECAO_PARA_CODIGO: dict[Direcao, int] = {
    Direcao.CENTRO: 0,
    Direcao.CIMA: 1,
    Direcao.BAIXO: 2,
    Direcao.ESQUERDA: 3,
    Direcao.DIREITA: 4,
    Direcao.CIMA_ESQUERDA: 5,
    Direcao.CIMA_DIREITA: 6,
    Direcao.BAIXO_ESQUERDA: 7,
    Direcao.BAIXO_DIREITA: 8,
}


def codigo_de_direcao(direcao: Direcao, apenas_cardeais: bool = False) -> int:
    """Converte a direção no código inteiro do protocolo.

    Com ``apenas_cardeais``, as diagonais são projetadas no eixo de maior
    excursão para manter compatibilidade com clientes antigos que só
    reconhecem ``0..4``.
    """
    codigo = DIRECAO_PARA_CODIGO[direcao]
    if apenas_cardeais and codigo > 4:
        return {5: 1, 6: 1, 7: 2, 8: 2}[codigo]
    return codigo


@dataclass(frozen=True)
class QuadroAnalisado:
    """Resultado de um frame, com as evidências intermediárias preservadas."""

    resultado: Optional[ResultadoOlhar]
    medidas: Optional[MedidasOculares]
    marginais: Optional[MarginaisCNN]
    caixa_olho: Optional[tuple[int, int, int, int]]

    @property
    def direcao(self) -> Direcao:
        return self.resultado.direcao if self.resultado else Direcao.CENTRO

    @property
    def rosto_detectado(self) -> bool:
        return self.resultado is not None


class PipelineOlhar:
    """Orquestra as etapas e mantém o estado temporal entre frames."""

    def __init__(
        self,
        pesos_cnn: str | Path = CAMINHO_PADRAO_PESOS,
        modelo_landmarker: str | Path = CAMINHO_PADRAO_LANDMARKER,
        config: Optional[ConfigFusao] = None,
        modo_video: bool = True,
        usar_cnn: bool = True,
        indices_olho: Sequence[int] = OLHO_DIREITO,
        indices_iris: Sequence[int] = IRIS_DIREITA,
        inverter_horizontal_cnn: bool = False,
        escala_de_cinza: bool = True,
        limite_memoria_gpu: float | None = None,
    ) -> None:
        self.detector = DetectorFacial(modelo_landmarker, modo_video=modo_video)
        self.classificador = (
            ClassificadorOlhar(
                pesos_cnn,
                inverter_horizontal=inverter_horizontal_cnn,
                escala_de_cinza=escala_de_cinza,
                limite_memoria_gpu=limite_memoria_gpu,
            )
            if usar_cnn
            else None
        )
        self.fusor = FusorParaconsistente(config)
        self.indices_olho = tuple(indices_olho)
        self.indices_iris = tuple(indices_iris)

    def __enter__(self) -> "PipelineOlhar":
        return self

    def __exit__(self, *_) -> None:
        self.fechar()

    def fechar(self) -> None:
        self.detector.fechar()

    def reiniciar(self) -> None:
        """Zera histerese e rastreamento — chamar entre vídeos ou sessões."""
        self.fusor.reiniciar()
        self.detector.reiniciar()

    def processar(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> QuadroAnalisado:
        """Analisa um frame BGR e devolve a direção com suas evidências."""
        deteccao = self.detector.detectar(
            frame_bgr,
            timestamp_ms=timestamp_ms,
            olho=self.indices_olho,
            iris=self.indices_iris,
        )
        if deteccao is None:
            return QuadroAnalisado(None, None, None, None)

        medidas = medir(deteccao.contorno, deteccao.iris)
        marginais = (
            self.classificador.prever(deteccao.recorte_rgb)
            if self.classificador is not None
            else None
        )
        resultado = self.fusor.inferir(marginais, medidas)

        return QuadroAnalisado(
            resultado=resultado,
            medidas=medidas,
            marginais=marginais,
            caixa_olho=deteccao.caixa,
        )
