"""EyeTeractive — rastreamento ocular assistivo de baixo custo.

Arquitetura: a CNN (ResNet-101) governa o eixo vertical, a geometria da íris
governa o eixo horizontal, e a lógica paraconsistente anotada evidencial
arbitra as duas fontes e produz as nove direções do olhar.
"""

from .fusion import (
    ConfigFusao,
    Direcao,
    EixoHorizontal,
    EixoVertical,
    FusorParaconsistente,
    LeituraEixo,
    ResultadoOlhar,
)
from .geometry import MedidasOculares, medir
from .paraconsistent import Anotacao, EstadoLogico

__version__ = "0.2.0"

__all__ = [
    "Anotacao",
    "ConfigFusao",
    "Direcao",
    "EixoHorizontal",
    "EixoVertical",
    "EstadoLogico",
    "FusorParaconsistente",
    "LeituraEixo",
    "MedidasOculares",
    "ResultadoOlhar",
    "medir",
    "__version__",
]
