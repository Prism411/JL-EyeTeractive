"""Medidas geométricas da íris a partir dos landmarks do contorno ocular.

Herda a formulação por *contribuição percentual de distâncias* do protótipo
original (``matrixHandler.py``), preservando a escala dos graus para manter
comparabilidade com os resultados já publicados, mas corrigindo os defeitos
que inviabilizavam o eixo vertical.

Convenção dos seis pontos do contorno (ordem obrigatória)::

    índice  landmark  posição
    0       33        canto externo   (lado temporal)
    1       160       pálpebra superior, próximo ao canto externo
    2       158       pálpebra superior, próximo ao canto interno
    3       133       canto interno   (lado nasal)
    4       153       pálpebra inferior, próximo ao canto interno
    5       144       pálpebra inferior, próximo ao canto externo

Para o olho **direito** da pessoa (que aparece à esquerda na imagem) os
índices ``{0, 1, 5}`` ficam do lado esquerdo da imagem e ``{2, 3, 4}`` do lado
direito. A distância da íris a um grupo cresce quando ela se afasta dele, de
modo que ``grau_horizontal > 0`` significa íris deslocada para a direita da
imagem e ``grau_vertical > 0`` significa íris deslocada para cima.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

__all__ = [
    "MedidasOculares",
    "ESCALA_HORIZONTAL",
    "ESCALA_VERTICAL",
    "centroide",
    "distancias_ao_ponto",
    "contribuicao_percentual",
    "graus",
    "abertura_relativa",
    "medir",
]

Ponto = Tuple[float, float]

# Grupos de índices por lado, derivados da convenção documentada acima.
_LADO_ESQUERDO = (0, 1, 5)
_LADO_DIREITO = (2, 3, 4)
_PALPEBRA_SUPERIOR = (1, 2)
_PALPEBRA_INFERIOR = (4, 5)

# Saturação empírica dos graus, em pontos percentuais. O eixo vertical satura
# muito antes porque a pálpebra acompanha o olhar e comprime o sinal.
ESCALA_HORIZONTAL = 25.0
ESCALA_VERTICAL = 8.0

# Abertura ocular (EAR) abaixo da qual a geometria vertical perde sentido:
# a pálpebra oclui a íris e as distâncias verticais deixam de medir o olhar.
EAR_MINIMO = 0.18
EAR_CONFIAVEL = 0.28


@dataclass(frozen=True)
class MedidasOculares:
    """Medidas geométricas de um único olho num frame."""

    grau_horizontal: float
    grau_vertical: float
    abertura: float
    confianca_vertical: float
    centroide_iris: Ponto


def centroide(pontos: Sequence[Ponto]) -> Ponto:
    """Centroide de um conjunto de pontos.

    Diferente do original, não arredonda para inteiro: o arredondamento
    descartava até meio pixel por eixo, o que é significativo num olho de
    ~30 px de largura.
    """
    if not pontos:
        raise ValueError("É preciso ao menos um ponto para calcular o centroide.")
    n = len(pontos)
    return (sum(p[0] for p in pontos) / n, sum(p[1] for p in pontos) / n)


def distancias_ao_ponto(pontos: Sequence[Ponto], referencia: Ponto) -> list[float]:
    """Distâncias euclidianas de cada ponto até a referência (sem arredondar)."""
    return [math.dist(p, referencia) for p in pontos]


def contribuicao_percentual(distancias: Sequence[float], indices: Sequence[int]) -> float:
    """Percentual da soma total de distâncias concentrado nos índices dados."""
    total = sum(distancias)
    if total <= 0:
        raise ValueError("A soma das distâncias deve ser positiva.")
    for i in indices:
        if not 0 <= i < len(distancias):
            raise IndexError(f"Índice {i} fora do intervalo de 'distancias'.")
    return sum(distancias[i] for i in indices) / total * 100.0


def graus(distancias: Sequence[float]) -> tuple[float, float]:
    """Graus horizontal e vertical, em pontos percentuais.

    ``grau_horizontal > 0`` → íris à direita da imagem.
    ``grau_vertical   > 0`` → íris para cima.

    O sinal do eixo vertical é o ponto corrigido em relação ao protótipo. Lá,
    ``grau_vertical`` era calculado como ``baixo − cima`` mas interpretado como
    "positivo é para cima", invertendo a resposta do eixo. Aqui o escore para
    cima é a contribuição da pálpebra *inferior* — a íris se afasta dela ao
    subir — e ele entra com sinal positivo.
    """
    if len(distancias) != 6:
        raise ValueError("São esperadas exatamente 6 distâncias do contorno ocular.")

    lado_esquerdo = contribuicao_percentual(distancias, _LADO_ESQUERDO)
    lado_direito = contribuicao_percentual(distancias, _LADO_DIREITO)
    grau_horizontal = lado_esquerdo - lado_direito

    escore_cima = contribuicao_percentual(distancias, _PALPEBRA_INFERIOR)
    escore_baixo = contribuicao_percentual(distancias, _PALPEBRA_SUPERIOR)
    grau_vertical = escore_cima - escore_baixo

    return grau_horizontal, grau_vertical


def abertura_relativa(pontos: Sequence[Ponto]) -> float:
    """Razão de aspecto do olho (EAR), invariante à escala do rosto.

    ``EAR = (‖p1−p5‖ + ‖p2−p4‖) / (2·‖p0−p3‖)`` na convenção documentada.
    Serve como medidor de oclusão palpebral: quanto menor, menos a geometria
    vertical tem a dizer sobre o olhar.
    """
    if len(pontos) != 6:
        raise ValueError("São esperados exatamente 6 pontos do contorno ocular.")

    largura = math.dist(pontos[0], pontos[3])
    if largura <= 0:
        return 0.0
    altura = math.dist(pontos[1], pontos[5]) + math.dist(pontos[2], pontos[4])
    return altura / (2.0 * largura)


def confianca_vertical(ear: float) -> float:
    """Confiança da geometria vertical em ``[0, 1]``, derivada da abertura.

    Abaixo de ``EAR_MINIMO`` a pálpebra cobre a íris e a confiança zera; acima
    de ``EAR_CONFIAVEL`` satura. Entre os dois, interpola linearmente.
    """
    if ear <= EAR_MINIMO:
        return 0.0
    if ear >= EAR_CONFIAVEL:
        return 1.0
    return (ear - EAR_MINIMO) / (EAR_CONFIAVEL - EAR_MINIMO)


def medir(contorno: Sequence[Ponto], iris: Sequence[Ponto]) -> MedidasOculares:
    """Pipeline geométrico completo para um olho."""
    centro_iris = centroide(iris)
    dists = distancias_ao_ponto(contorno, centro_iris)
    grau_h, grau_v = graus(dists)
    ear = abertura_relativa(contorno)
    return MedidasOculares(
        grau_horizontal=grau_h,
        grau_vertical=grau_v,
        abertura=ear,
        confianca_vertical=confianca_vertical(ear),
        centroide_iris=centro_iris,
    )
