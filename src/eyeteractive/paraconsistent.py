"""Lógica Paraconsistente Anotada Evidencial Eτ.

Implementa o para-analisador de Da Costa/Abe: cada proposição recebe uma
anotação ``(μ, λ)`` onde ``μ`` é o grau de evidência favorável e ``λ`` o grau
de evidência contrária, ambos em ``[0, 1]``.

A partir da anotação derivam-se:

- **Grau de certeza** ``Gc = μ − λ`` — o quanto a proposição pende para
  verdadeira (``+1``) ou falsa (``−1``).
- **Grau de contradição** ``Gct = μ + λ − 1`` — positivo quando as fontes
  afirmam *e* negam ao mesmo tempo (inconsistência), negativo quando nenhuma
  fonte se compromete (paracompletude/ignorância).

O ponto ``(Gc, Gct)`` vive no quadrado unitário girado 45° ("reticulado de
Hasse"). O **grau de certeza real** ``Gcr`` desconta a certeza pela distância
ao vértice verdadeiro/falso, de modo que evidência contraditória não é lida
como confiança alta — é exatamente isso que a cadeia de ``if/elif`` do código
original não fazia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

__all__ = [
    "Anotacao",
    "EstadoLogico",
    "combinar",
    "maximizacao",
    "minimizacao",
    "evidencia_de_escore",
    "evidencia_de_probabilidades",
]


def _limitar(valor: float, minimo: float = 0.0, maximo: float = 1.0) -> float:
    return max(minimo, min(maximo, valor))


class EstadoLogico(str, Enum):
    """Estados de saída do para-analisador."""

    VERDADEIRO = "verdadeiro"
    FALSO = "falso"
    INCONSISTENTE = "inconsistente"  # ⊤ — fontes se contradizem
    PARACOMPLETO = "paracompleto"  # ⊥ — nenhuma fonte se compromete
    INDETERMINADO = "indeterminado"  # região não-extrema do reticulado

    @property
    def conclusivo(self) -> bool:
        """Verdadeiro quando o estado permite decidir a proposição."""
        return self in (EstadoLogico.VERDADEIRO, EstadoLogico.FALSO)


@dataclass(frozen=True)
class Anotacao:
    """Anotação evidencial ``(μ, λ)`` de uma proposição."""

    mu: float
    lam: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu", _limitar(float(self.mu)))
        object.__setattr__(self, "lam", _limitar(float(self.lam)))

    @property
    def certeza(self) -> float:
        """Grau de certeza ``Gc = μ − λ`` em ``[−1, 1]``."""
        return self.mu - self.lam

    @property
    def contradicao(self) -> float:
        """Grau de contradição ``Gct = μ + λ − 1`` em ``[−1, 1]``."""
        return self.mu + self.lam - 1.0

    @property
    def certeza_real(self) -> float:
        """Certeza descontada pela contradição (``Gcr``).

        Mede o quanto o ponto ``(Gc, Gct)`` se aproxima do vértice
        verdadeiro (``Gc = 1``) ou falso (``Gc = −1``). Sob contradição alta o
        valor colapsa para zero mesmo que ``Gc`` seja grande — é este número,
        e não ``Gc``, que deve ser reportado como confiança.
        """
        gc = self.certeza
        distancia = math.hypot(1.0 - abs(gc), self.contradicao)
        if distancia >= 1.0:
            return 0.0
        magnitude = 1.0 - distancia
        return math.copysign(magnitude, gc) if gc != 0.0 else 0.0

    def estado(self, limiar_certeza: float = 0.30, limiar_contradicao: float = 0.55) -> EstadoLogico:
        """Classifica a anotação nas regiões do reticulado.

        A contradição é avaliada antes da certeza: sob inconsistência ou
        paracompletude a decisão é recusada mesmo que ``Gc`` esteja alto.
        """
        gct = self.contradicao
        if gct >= limiar_contradicao:
            return EstadoLogico.INCONSISTENTE
        if gct <= -limiar_contradicao:
            return EstadoLogico.PARACOMPLETO

        gc = self.certeza
        if gc >= limiar_certeza:
            return EstadoLogico.VERDADEIRO
        if gc <= -limiar_certeza:
            return EstadoLogico.FALSO
        return EstadoLogico.INDETERMINADO

    def negada(self) -> "Anotacao":
        """Negação paraconsistente: troca evidência favorável e contrária."""
        return Anotacao(self.lam, self.mu)

    def __repr__(self) -> str:  # pragma: no cover - conveniência de depuração
        return (
            f"Anotacao(mu={self.mu:.3f}, lam={self.lam:.3f}, "
            f"Gc={self.certeza:+.3f}, Gct={self.contradicao:+.3f})"
        )


def combinar(anotacoes: Sequence[Anotacao], pesos: Sequence[float] | None = None) -> Anotacao:
    """Célula de conexão evidencial: média ponderada de várias fontes.

    Preserva a contradição entre as fontes — se uma afirma (``μ=0,9``) e outra
    nega (``μ=0,1``), o resultado tem ``Gc ≈ 0`` e ``Gct`` alto, sinalizando o
    conflito em vez de escondê-lo numa média silenciosa.
    """
    if not anotacoes:
        raise ValueError("É preciso ao menos uma anotação para combinar.")

    if pesos is None:
        pesos = [1.0] * len(anotacoes)
    if len(pesos) != len(anotacoes):
        raise ValueError("pesos e anotacoes devem ter o mesmo comprimento.")

    total = sum(pesos)
    if total <= 0:
        raise ValueError("A soma dos pesos deve ser positiva.")

    mu = sum(a.mu * p for a, p in zip(anotacoes, pesos)) / total
    lam = sum(a.lam * p for a, p in zip(anotacoes, pesos)) / total
    return Anotacao(mu, lam)


def maximizacao(anotacoes: Iterable[Anotacao]) -> Anotacao:
    """Operador OR de Eτ: ``μ = max(μᵢ)``, ``λ = min(λᵢ)``."""
    itens = list(anotacoes)
    if not itens:
        raise ValueError("É preciso ao menos uma anotação.")
    return Anotacao(max(a.mu for a in itens), min(a.lam for a in itens))


def minimizacao(anotacoes: Iterable[Anotacao]) -> Anotacao:
    """Operador AND de Eτ: ``μ = min(μᵢ)``, ``λ = max(λᵢ)``."""
    itens = list(anotacoes)
    if not itens:
        raise ValueError("É preciso ao menos uma anotação.")
    return Anotacao(min(a.mu for a in itens), max(a.lam for a in itens))


def evidencia_de_escore(escore: float, escala: float, confianca: float = 1.0) -> Anotacao:
    """Converte um escore bipolar contínuo em anotação evidencial.

    ``escore`` positivo sustenta a proposição, negativo a refuta; ``escala`` é
    o valor de saturação. ``confianca ∈ [0, 1]`` reduz *ambas* as evidências,
    empurrando a anotação para a região paracompleta quando a fonte é pouco
    confiável — o mecanismo que permite rebaixar a geometria vertical sem
    descartá-la.
    """
    if escala <= 0:
        raise ValueError("escala deve ser positiva.")

    normalizado = _limitar(escore / escala, -1.0, 1.0)
    confianca = _limitar(confianca)

    favoravel = (1.0 + normalizado) / 2.0
    return Anotacao(favoravel * confianca, (1.0 - favoravel) * confianca)


def evidencia_de_probabilidades(
    favoravel: float, contraria: float, confianca: float = 1.0
) -> Anotacao:
    """Monta a anotação a partir de massas de probabilidade concorrentes.

    Usada para transformar marginais de softmax em ``(μ, λ)``. A massa que não
    apoia nem refuta (por exemplo, a classe *centro* ao decidir "está para
    cima?") permanece fora, deixando a anotação deliberadamente paracompleta.
    """
    favoravel = _limitar(favoravel)
    contraria = _limitar(contraria)
    confianca = _limitar(confianca)
    return Anotacao(favoravel * confianca, contraria * confianca)
