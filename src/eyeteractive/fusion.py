"""Fusão paraconsistente das fontes de evidência em nove direções do olhar.

Divisão de responsabilidades, medida e não arbitrada:

**Eixo vertical — a CNN decide.** Medido em vídeo de olhar para cima, o grau
vertical geométrico tem excursão de poucos pontos percentuais: menor que a
própria banda neutra que separaria "cima" de "centro". A pálpebra acompanha o
olhar e comprime a excursão da íris, então nenhuma calibração de limiar
recupera essa resolução. A geometria vertical permanece no sistema, mas com
peso pequeno e como fonte de contradição — serve para acusar desacordo, não
para decidir.

**Eixo horizontal — a geometria decide.** Ali a excursão é ampla (dezenas de
pontos percentuais), o cálculo é interpretável e não depende de GPU. A
marginal horizontal da rede entra como segunda fonte de peso comparável.

**A lógica paraconsistente arbitra.** Cada eixo vira uma proposição bipolar
anotada ``(μ, λ)``: para o vertical, "o olhar está para cima", com ``μ`` vindo
da evidência de cima e ``λ`` da de baixo. As fontes são combinadas numa célula
de conexão evidencial e a decisão sai do **grau de certeza real** ``Gcr``, que
desconta a certeza pela contradição. Quando as fontes se contradizem, ``Gcr``
colapsa e o eixo cai em "centro" em vez de oscilar — comportamento que a
cadeia de ``if/elif`` do protótipo não tinha como produzir.

As nove direções são o produto cartesiano dos dois eixos ternários.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Optional

from .cnn import MarginaisCNN
from .geometry import ESCALA_HORIZONTAL, ESCALA_VERTICAL, MedidasOculares
from .paraconsistent import (
    Anotacao,
    EstadoLogico,
    combinar,
    evidencia_de_escore,
    evidencia_de_probabilidades,
)

__all__ = [
    "Direcao",
    "EixoHorizontal",
    "EixoVertical",
    "LeituraEixo",
    "ResultadoOlhar",
    "ConfigFusao",
    "FusorParaconsistente",
]


class EixoVertical(str, Enum):
    CIMA = "cima"
    CENTRO = "centro"
    BAIXO = "baixo"


class EixoHorizontal(str, Enum):
    ESQUERDA = "esquerda"
    CENTRO = "centro"
    DIREITA = "direita"


class Direcao(str, Enum):
    """As nove direções resultantes do cruzamento dos dois eixos."""

    CENTRO = "centro"
    CIMA = "cima"
    BAIXO = "baixo"
    ESQUERDA = "esquerda"
    DIREITA = "direita"
    CIMA_ESQUERDA = "cima-esquerda"
    CIMA_DIREITA = "cima-direita"
    BAIXO_ESQUERDA = "baixo-esquerda"
    BAIXO_DIREITA = "baixo-direita"

    @classmethod
    def compor(cls, vertical: EixoVertical, horizontal: EixoHorizontal) -> "Direcao":
        if vertical is EixoVertical.CENTRO and horizontal is EixoHorizontal.CENTRO:
            return cls.CENTRO
        if vertical is EixoVertical.CENTRO:
            return cls(horizontal.value)
        if horizontal is EixoHorizontal.CENTRO:
            return cls(vertical.value)
        return cls(f"{vertical.value}-{horizontal.value}")

    def eixos(self) -> tuple[EixoVertical, EixoHorizontal]:
        """Decompõe a direção nos dois eixos — inverso de :meth:`compor`."""
        if self is Direcao.CENTRO:
            return EixoVertical.CENTRO, EixoHorizontal.CENTRO
        if "-" in self.value:
            vertical, horizontal = self.value.split("-", 1)
            return EixoVertical(vertical), EixoHorizontal(horizontal)
        if self.value in {e.value for e in EixoVertical}:
            return EixoVertical(self.value), EixoHorizontal.CENTRO
        return EixoVertical.CENTRO, EixoHorizontal(self.value)


@dataclass(frozen=True)
class LeituraEixo:
    """Decisão de um eixo com a evidência que a sustenta."""

    valor: str
    anotacao: Anotacao
    estado: EstadoLogico

    @property
    def certeza_real(self) -> float:
        return self.anotacao.certeza_real

    @property
    def contraditorio(self) -> bool:
        return self.estado is EstadoLogico.INCONSISTENTE


@dataclass(frozen=True)
class ResultadoOlhar:
    """Saída completa de um frame.

    Traz as duas leituras porque elas servem a propósitos diferentes:
    ``direcao`` é a decisão já filtrada pela histerese, que deve governar o
    comando enviado ao dispositivo, e ``direcao_instantanea`` é o que este
    frame isolado diz. Elas divergem durante uma transição — inclusive no
    primeiro frame de uma sessão, quando a estável ainda é ``CENTRO``.
    """

    direcao: Direcao
    direcao_instantanea: Direcao
    vertical: LeituraEixo
    horizontal: LeituraEixo
    confianca: float
    estavel: bool

    def resumo(self) -> str:
        transicao = (
            "" if self.estavel else f"  (transitório → {self.direcao_instantanea.value})"
        )
        return (
            f"{self.direcao.value:<15} conf={self.confianca:.2f} "
            f"V[{self.vertical.valor} Gcr={self.vertical.certeza_real:+.2f} "
            f"{self.vertical.estado.value}] "
            f"H[{self.horizontal.valor} Gcr={self.horizontal.certeza_real:+.2f} "
            f"{self.horizontal.estado.value}]"
            + transicao
        )


@dataclass(frozen=True)
class ConfigFusao:
    """Parâmetros da fusão, todos derivados de medição.

    Tudo aqui é calibrável por usuário: a excursão do grau horizontal depende
    do formato do olho e da distância à câmera, e os limiares de decisão
    determinam o quanto o sistema exige de evidência antes de emitir um
    comando — uma escolha que troca sensibilidade por estabilidade e que
    diferentes usuários resolvem de formas diferentes.

    Use :mod:`eyeteractive.calibracao` para ajustá-los a partir de uma sessão
    de calibração e ``de_dicionario``/``para_dicionario`` para persistir.
    """

    # Decisão sai do grau de certeza real, já descontado pela contradição.
    limiar_decisao_vertical: float = 0.15
    limiar_decisao_horizontal: float = 0.18

    # Saturação dos graus geométricos, em pontos percentuais.
    escala_horizontal: float = ESCALA_HORIZONTAL
    escala_vertical: float = ESCALA_VERTICAL

    # Classificação de estado no reticulado.
    limiar_certeza: float = 0.30
    limiar_contradicao: float = 0.55

    # Pesos das fontes por eixo.
    peso_cnn_vertical: float = 1.0
    peso_geometria_vertical: float = 0.15  # só acusa contradição; não decide
    peso_cnn_horizontal: float = 0.5
    peso_geometria_horizontal: float = 1.0

    # Histerese temporal: quantos frames consecutivos confirmam uma mudança.
    # Use ``1`` para desligá-la — é o valor certo para análise de imagens
    # avulsas, em que não existe continuidade temporal a explorar.
    frames_para_mudar: int = 2

    def para_dicionario(self) -> dict[str, float | int]:
        """Serializa para JSON, para gravar um perfil de usuário."""
        return asdict(self)

    @classmethod
    def de_dicionario(cls, dados: dict) -> "ConfigFusao":
        """Reconstrói a partir de um perfil, ignorando chaves desconhecidas.

        Perfis gravados por versões anteriores continuam carregáveis; campos
        ausentes assumem o padrão.
        """
        conhecidos = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dados.items() if k in conhecidos})


class FusorParaconsistente:
    """Combina CNN e geometria numa das nove direções, com histerese."""

    def __init__(self, config: Optional[ConfigFusao] = None) -> None:
        self.config = config or ConfigFusao()
        self._direcao_estavel: Direcao = Direcao.CENTRO
        self._candidata: Optional[Direcao] = None
        self._contagem_candidata = 0

    def reiniciar(self) -> None:
        """Zera o estado temporal — chamar ao trocar de vídeo ou de usuário."""
        self._direcao_estavel = Direcao.CENTRO
        self._candidata = None
        self._contagem_candidata = 0

    # ------------------------------------------------------------------
    # Montagem das anotações por eixo
    # ------------------------------------------------------------------

    def _anotacao_vertical(
        self, marginais: MarginaisCNN, medidas: Optional[MedidasOculares]
    ) -> Anotacao:
        """Proposição: "o olhar está para CIMA"."""
        cfg = self.config
        fontes = [evidencia_de_probabilidades(marginais.cima, marginais.baixo)]
        pesos = [cfg.peso_cnn_vertical]

        if medidas is not None and cfg.peso_geometria_vertical > 0:
            fontes.append(
                evidencia_de_escore(
                    medidas.grau_vertical,
                    cfg.escala_vertical,
                    confianca=medidas.confianca_vertical,
                )
            )
            pesos.append(cfg.peso_geometria_vertical)

        return combinar(fontes, pesos)

    def _anotacao_horizontal(
        self, marginais: Optional[MarginaisCNN], medidas: MedidasOculares
    ) -> Anotacao:
        """Proposição: "o olhar está para a DIREITA"."""
        cfg = self.config
        fontes = [evidencia_de_escore(medidas.grau_horizontal, cfg.escala_horizontal)]
        pesos = [cfg.peso_geometria_horizontal]

        if marginais is not None and cfg.peso_cnn_horizontal > 0:
            fontes.append(
                evidencia_de_probabilidades(marginais.direita, marginais.esquerda)
            )
            pesos.append(cfg.peso_cnn_horizontal)

        return combinar(fontes, pesos)

    # ------------------------------------------------------------------
    # Decisão
    # ------------------------------------------------------------------

    def _decidir(
        self, anotacao: Anotacao, limiar: float, positivo: str, negativo: str
    ) -> LeituraEixo:
        cfg = self.config
        estado = anotacao.estado(cfg.limiar_certeza, cfg.limiar_contradicao)

        # Sob inconsistência a certeza real já colapsou; a decisão cai em
        # centro por construção, sem precisar de ramo especial.
        gcr = anotacao.certeza_real
        if gcr >= limiar:
            valor = positivo
        elif gcr <= -limiar:
            valor = negativo
        else:
            valor = "centro"

        return LeituraEixo(valor=valor, anotacao=anotacao, estado=estado)

    def inferir(
        self,
        marginais: Optional[MarginaisCNN],
        medidas: Optional[MedidasOculares],
    ) -> ResultadoOlhar:
        """Funde as fontes disponíveis numa direção.

        Aceita fontes ausentes: sem CNN o eixo vertical fica paracompleto (e
        portanto em "centro"), sem geometria o horizontal se apoia só na rede.
        """
        if marginais is None and medidas is None:
            raise ValueError("É preciso ao menos uma fonte de evidência.")

        if marginais is not None:
            anotacao_v = self._anotacao_vertical(marginais, medidas)
        else:
            anotacao_v = Anotacao(0.0, 0.0)  # paracompleto: nada a afirmar

        if medidas is not None:
            anotacao_h = self._anotacao_horizontal(marginais, medidas)
        elif marginais is not None:
            anotacao_h = evidencia_de_probabilidades(marginais.direita, marginais.esquerda)
        else:  # pragma: no cover - coberto pela validação acima
            anotacao_h = Anotacao(0.0, 0.0)

        leitura_v = self._decidir(
            anotacao_v, self.config.limiar_decisao_vertical, "cima", "baixo"
        )
        leitura_h = self._decidir(
            anotacao_h, self.config.limiar_decisao_horizontal, "direita", "esquerda"
        )

        direcao = Direcao.compor(
            EixoVertical(leitura_v.valor), EixoHorizontal(leitura_h.valor)
        )
        estavel = self._aplicar_histerese(direcao)

        confianca = min(abs(leitura_v.certeza_real), abs(leitura_h.certeza_real)) if (
            leitura_v.valor != "centro" and leitura_h.valor != "centro"
        ) else max(abs(leitura_v.certeza_real), abs(leitura_h.certeza_real))

        return ResultadoOlhar(
            direcao=self._direcao_estavel,
            direcao_instantanea=direcao,
            vertical=leitura_v,
            horizontal=leitura_h,
            confianca=confianca,
            estavel=estavel,
        )

    def _aplicar_histerese(self, direcao: Direcao) -> bool:
        """Só troca a direção após ``frames_para_mudar`` confirmações seguidas.

        Devolve ``True`` quando o frame atual concorda com a direção estável.
        """
        if direcao == self._direcao_estavel:
            self._candidata = None
            self._contagem_candidata = 0
            return True

        if direcao == self._candidata:
            self._contagem_candidata += 1
        else:
            self._candidata = direcao
            self._contagem_candidata = 1

        if self._contagem_candidata >= self.config.frames_para_mudar:
            self._direcao_estavel = direcao
            self._candidata = None
            self._contagem_candidata = 0
            return True

        return False
