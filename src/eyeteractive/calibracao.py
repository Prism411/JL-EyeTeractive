"""Calibração por usuário dos parâmetros de fusão.

Os padrões de :class:`~eyeteractive.fusion.ConfigFusao` foram fixados a partir
dos vídeos de um único participante. Dois fatores fazem eles não transferirem:

- **A excursão do grau horizontal depende da anatomia e do enquadramento.** Um
  olho mais alongado, ou uma câmera mais distante, produz percentuais menores
  para o mesmo ângulo de olhar. Uma escala mal ajustada satura cedo demais (o
  sistema vira "sempre esquerda/direita") ou nunca (o sistema vira "sempre
  centro").
- **O limiar de decisão é uma troca entre sensibilidade e estabilidade**, e o
  ponto de equilíbrio depende de quanto o usuário consegue fixar o olhar.

A calibração coleta amostras rotuladas — o usuário olha para alvos conhecidos —
e escolhe os parâmetros por busca em grade, maximizando o acerto por eixo. Não
há retreino da rede: só se ajustam os parâmetros da camada de decisão, o que
mantém o procedimento curto o bastante para ser feito no início de uma sessão.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .cnn import MarginaisCNN
from .fusion import ConfigFusao, FusorParaconsistente
from .geometry import MedidasOculares

__all__ = [
    "AmostraCalibracao",
    "ResultadoCalibracao",
    "ALVOS",
    "acuracia_por_eixo",
    "calibrar",
    "salvar_perfil",
    "carregar_perfil",
]

# Alvos mínimos de uma sessão: cada eixo precisa dos três estados.
ALVOS: tuple[tuple[str, str, str], ...] = (
    ("centro", "centro", "centro"),
    ("cima", "cima", "centro"),
    ("baixo", "baixo", "centro"),
    ("esquerda", "centro", "esquerda"),
    ("direita", "centro", "direita"),
)


@dataclass(frozen=True)
class AmostraCalibracao:
    """Uma observação rotulada: evidências e o alvo que o usuário fixava."""

    marginais: Optional[MarginaisCNN]
    medidas: Optional[MedidasOculares]
    vertical_esperado: str
    horizontal_esperado: str


@dataclass(frozen=True)
class ResultadoCalibracao:
    """Configuração escolhida e o desempenho que a justificou."""

    config: ConfigFusao
    acuracia_vertical: float
    acuracia_horizontal: float
    acuracia_vertical_padrao: float
    acuracia_horizontal_padrao: float
    n_amostras: int

    @property
    def ganho_vertical(self) -> float:
        return self.acuracia_vertical - self.acuracia_vertical_padrao

    @property
    def ganho_horizontal(self) -> float:
        return self.acuracia_horizontal - self.acuracia_horizontal_padrao

    def resumo(self) -> str:
        return (
            f"vertical   {self.acuracia_vertical_padrao * 100:5.1f}% → "
            f"{self.acuracia_vertical * 100:5.1f}%  ({self.ganho_vertical * 100:+.1f} pp)\n"
            f"horizontal {self.acuracia_horizontal_padrao * 100:5.1f}% → "
            f"{self.acuracia_horizontal * 100:5.1f}%  ({self.ganho_horizontal * 100:+.1f} pp)"
        )


def acuracia_por_eixo(
    amostras: Sequence[AmostraCalibracao], config: ConfigFusao
) -> tuple[float, float]:
    """Acerto de cada eixo sob uma configuração, sem histerese.

    A histerese é desligada de propósito: ela mede estabilidade temporal, não
    correção da decisão, e amostras de calibração são independentes.
    """
    if not amostras:
        return 0.0, 0.0

    sem_histerese = ConfigFusao(**{**config.para_dicionario(), "frames_para_mudar": 1})
    fusor = FusorParaconsistente(sem_histerese)

    certos_v = certos_h = 0
    for amostra in amostras:
        fusor.reiniciar()
        resultado = fusor.inferir(amostra.marginais, amostra.medidas)
        certos_v += resultado.vertical.valor == amostra.vertical_esperado
        certos_h += resultado.horizontal.valor == amostra.horizontal_esperado

    n = len(amostras)
    return certos_v / n, certos_h / n


def _grade(inicio: float, fim: float, passos: int) -> list[float]:
    if passos < 2:
        return [inicio]
    incremento = (fim - inicio) / (passos - 1)
    return [inicio + i * incremento for i in range(passos)]


def calibrar(
    amostras: Sequence[AmostraCalibracao],
    base: Optional[ConfigFusao] = None,
    limiares: Optional[Iterable[float]] = None,
    escalas_horizontais: Optional[Iterable[float]] = None,
    escalas_verticais: Optional[Iterable[float]] = None,
) -> ResultadoCalibracao:
    """Escolhe os parâmetros por busca em grade sobre as amostras rotuladas.

    Os eixos são otimizados separadamente porque são decididos por fontes
    diferentes e não competem por parâmetros: o vertical depende do limiar e da
    escala vertical, o horizontal dos seus próprios.

    Empates são resolvidos pelo limiar mais alto, que exige mais evidência
    antes de emitir um comando — em uso assistivo, um comando errado custa mais
    caro que um comando ausente.
    """
    if not amostras:
        raise ValueError("É preciso ao menos uma amostra para calibrar.")

    base = base or ConfigFusao()
    limiares = list(limiares) if limiares is not None else _grade(0.05, 0.60, 12)
    escalas_h = (
        list(escalas_horizontais)
        if escalas_horizontais is not None
        else _grade(8.0, 40.0, 9)
    )
    # A excursão vertical medida é de ~±3,5 pontos percentuais, então a grade
    # precisa descer bem abaixo disso para não travar no limite inferior.
    escalas_v = (
        list(escalas_verticais) if escalas_verticais is not None else _grade(1.5, 15.0, 10)
    )

    padrao_v, padrao_h = acuracia_por_eixo(amostras, base)

    melhor_v = (padrao_v, base.limiar_decisao_vertical, base.escala_vertical)
    for escala in escalas_v:
        for limiar in limiares:
            config = ConfigFusao(**{
                **base.para_dicionario(),
                "limiar_decisao_vertical": limiar,
                "escala_vertical": escala,
            })
            acerto, _ = acuracia_por_eixo(amostras, config)
            if (acerto, limiar) > (melhor_v[0], melhor_v[1]):
                melhor_v = (acerto, limiar, escala)

    melhor_h = (padrao_h, base.limiar_decisao_horizontal, base.escala_horizontal)
    for escala in escalas_h:
        for limiar in limiares:
            config = ConfigFusao(**{
                **base.para_dicionario(),
                "limiar_decisao_horizontal": limiar,
                "escala_horizontal": escala,
            })
            _, acerto = acuracia_por_eixo(amostras, config)
            if (acerto, limiar) > (melhor_h[0], melhor_h[1]):
                melhor_h = (acerto, limiar, escala)

    final = ConfigFusao(**{
        **base.para_dicionario(),
        "limiar_decisao_vertical": melhor_v[1],
        "escala_vertical": melhor_v[2],
        "limiar_decisao_horizontal": melhor_h[1],
        "escala_horizontal": melhor_h[2],
    })

    return ResultadoCalibracao(
        config=final,
        acuracia_vertical=melhor_v[0],
        acuracia_horizontal=melhor_h[0],
        acuracia_vertical_padrao=padrao_v,
        acuracia_horizontal_padrao=padrao_h,
        n_amostras=len(amostras),
    )


def salvar_perfil(caminho: str | Path, resultado: ResultadoCalibracao, usuario: str = "") -> None:
    """Grava o perfil calibrado, com as métricas que o justificam."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(
            {
                "usuario": usuario,
                "n_amostras": resultado.n_amostras,
                "acuracia_vertical": resultado.acuracia_vertical,
                "acuracia_horizontal": resultado.acuracia_horizontal,
                "acuracia_vertical_padrao": resultado.acuracia_vertical_padrao,
                "acuracia_horizontal_padrao": resultado.acuracia_horizontal_padrao,
                "config": resultado.config.para_dicionario(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def carregar_perfil(caminho: str | Path) -> ConfigFusao:
    """Lê um perfil gravado e devolve a configuração de fusão."""
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return ConfigFusao.de_dicionario(dados.get("config", dados))
