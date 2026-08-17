"""Testes da fusão paraconsistente e da composição das nove direções."""

from __future__ import annotations

import pytest

from eyeteractive.cnn import marginalizar
from eyeteractive.fusion import (
    ConfigFusao,
    Direcao,
    EixoHorizontal,
    EixoVertical,
    FusorParaconsistente,
)
from eyeteractive.geometry import MedidasOculares

# Ordem das classes: center, down, left, right, up.
SOFTMAX_CIMA = (0.02, 0.01, 0.02, 0.01, 0.94)
SOFTMAX_BAIXO = (0.02, 0.94, 0.02, 0.01, 0.01)
SOFTMAX_CENTRO = (0.94, 0.02, 0.01, 0.02, 0.01)
SOFTMAX_CIMA_ESQUERDA = (0.03, 0.01, 0.48, 0.02, 0.46)


def medidas(grau_h: float, grau_v: float = 0.0, confianca: float = 1.0) -> MedidasOculares:
    return MedidasOculares(
        grau_horizontal=grau_h,
        grau_vertical=grau_v,
        abertura=0.30,
        confianca_vertical=confianca,
        centroide_iris=(0.0, 0.0),
    )


def assentar(fusor: FusorParaconsistente, marginais, med, repeticoes: int = 4):
    """Roda frames idênticos até a histerese assentar e devolve o último."""
    resultado = None
    for _ in range(repeticoes):
        resultado = fusor.inferir(marginais, med)
    return resultado


# ----------------------------------------------------------------------
# Composição das nove direções
# ----------------------------------------------------------------------


def test_compor_cobre_as_nove_direcoes():
    combinacoes = {
        Direcao.compor(v, h) for v in EixoVertical for h in EixoHorizontal
    }
    assert combinacoes == set(Direcao)
    assert len(combinacoes) == 9


def test_eixos_inverte_compor_para_todas_as_direcoes():
    for direcao in Direcao:
        vertical, horizontal = direcao.eixos()
        assert Direcao.compor(vertical, horizontal) is direcao


def test_eixos_casos_nomeados():
    assert Direcao.CENTRO.eixos() == (EixoVertical.CENTRO, EixoHorizontal.CENTRO)
    assert Direcao.CIMA.eixos() == (EixoVertical.CIMA, EixoHorizontal.CENTRO)
    assert Direcao.ESQUERDA.eixos() == (EixoVertical.CENTRO, EixoHorizontal.ESQUERDA)
    assert Direcao.BAIXO_DIREITA.eixos() == (EixoVertical.BAIXO, EixoHorizontal.DIREITA)


def test_compor_casos_nomeados():
    assert Direcao.compor(EixoVertical.CENTRO, EixoHorizontal.CENTRO) is Direcao.CENTRO
    assert Direcao.compor(EixoVertical.CIMA, EixoHorizontal.CENTRO) is Direcao.CIMA
    assert Direcao.compor(EixoVertical.CENTRO, EixoHorizontal.ESQUERDA) is Direcao.ESQUERDA
    assert Direcao.compor(EixoVertical.CIMA, EixoHorizontal.ESQUERDA) is Direcao.CIMA_ESQUERDA


# ----------------------------------------------------------------------
# A CNN governa o eixo vertical
# ----------------------------------------------------------------------


def test_cnn_decide_cima_mesmo_com_geometria_muda():
    """Geometria dentro da zona morta não pode impedir a rede de decidir."""
    fusor = FusorParaconsistente()
    resultado = assentar(fusor, marginalizar(SOFTMAX_CIMA), medidas(grau_h=0.0, grau_v=0.0))
    assert resultado.vertical.valor == "cima"
    assert resultado.direcao is Direcao.CIMA


def test_cnn_decide_baixo():
    fusor = FusorParaconsistente()
    resultado = assentar(fusor, marginalizar(SOFTMAX_BAIXO), medidas(grau_h=0.0))
    assert resultado.vertical.valor == "baixo"


def test_geometria_vertical_nao_derruba_a_cnn():
    """Mesmo com a geometria apontando o oposto, o peso dela não decide."""
    fusor = FusorParaconsistente()
    resultado = assentar(
        fusor, marginalizar(SOFTMAX_CIMA), medidas(grau_h=0.0, grau_v=-8.0)
    )
    assert resultado.vertical.valor == "cima"


def test_sem_cnn_o_eixo_vertical_fica_paracompleto():
    fusor = FusorParaconsistente()
    resultado = assentar(fusor, None, medidas(grau_h=20.0, grau_v=3.4))
    assert resultado.vertical.valor == "centro"
    assert resultado.horizontal.valor == "direita"


# ----------------------------------------------------------------------
# A geometria governa o eixo horizontal
# ----------------------------------------------------------------------


def test_geometria_decide_horizontal():
    fusor = FusorParaconsistente()
    assert assentar(fusor, None, medidas(grau_h=25.0)).horizontal.valor == "direita"

    fusor.reiniciar()
    assert assentar(fusor, None, medidas(grau_h=-25.0)).horizontal.valor == "esquerda"

    fusor.reiniciar()
    assert assentar(fusor, None, medidas(grau_h=0.0)).horizontal.valor == "centro"


def test_diagonal_emerge_do_softmax_repartido():
    """Olhar diagonal reparte a massa entre 'up' e 'left' — e vira diagonal."""
    fusor = FusorParaconsistente()
    resultado = assentar(
        fusor, marginalizar(SOFTMAX_CIMA_ESQUERDA), medidas(grau_h=-22.0)
    )
    assert resultado.direcao is Direcao.CIMA_ESQUERDA


# ----------------------------------------------------------------------
# Comportamento paraconsistente
# ----------------------------------------------------------------------


def test_fontes_horizontais_em_conflito_caem_para_centro():
    """Geometria diz direita, rede diz esquerda: recusa em vez de escolher."""
    config = ConfigFusao(peso_cnn_horizontal=1.0, peso_geometria_horizontal=1.0)
    fusor = FusorParaconsistente(config)
    marginais = marginalizar((0.02, 0.01, 0.94, 0.01, 0.02))  # rede: esquerda
    resultado = assentar(fusor, marginais, medidas(grau_h=25.0))  # geometria: direita

    assert resultado.horizontal.valor == "centro"
    assert abs(resultado.horizontal.certeza_real) < 0.1


def test_centro_confiante_nao_e_lido_como_contradicao():
    fusor = FusorParaconsistente()
    resultado = assentar(fusor, marginalizar(SOFTMAX_CENTRO), medidas(grau_h=0.0))
    assert resultado.direcao is Direcao.CENTRO
    assert not resultado.vertical.contraditorio


def test_olho_ocluido_anula_a_contribuicao_geometrica():
    fusor = FusorParaconsistente()
    resultado = assentar(
        fusor, marginalizar(SOFTMAX_CIMA), medidas(grau_h=0.0, grau_v=8.0, confianca=0.0)
    )
    assert resultado.vertical.valor == "cima"


def test_inferir_exige_alguma_fonte():
    with pytest.raises(ValueError):
        FusorParaconsistente().inferir(None, None)


# ----------------------------------------------------------------------
# Histerese temporal
# ----------------------------------------------------------------------


def test_histerese_ignora_frame_isolado():
    """Um frame divergente não pode trocar o comando enviado ao dispositivo."""
    fusor = FusorParaconsistente(ConfigFusao(frames_para_mudar=3))
    assentar(fusor, marginalizar(SOFTMAX_CENTRO), medidas(0.0), repeticoes=3)

    ruido = fusor.inferir(marginalizar(SOFTMAX_CIMA), medidas(0.0))
    assert ruido.direcao is Direcao.CENTRO
    assert not ruido.estavel


def test_histerese_aceita_mudanca_sustentada():
    fusor = FusorParaconsistente(ConfigFusao(frames_para_mudar=3))
    assentar(fusor, marginalizar(SOFTMAX_CENTRO), medidas(0.0), repeticoes=3)

    for _ in range(3):
        resultado = fusor.inferir(marginalizar(SOFTMAX_CIMA), medidas(0.0))
    assert resultado.direcao is Direcao.CIMA
    assert resultado.estavel


def test_reiniciar_volta_ao_centro():
    fusor = FusorParaconsistente()
    assentar(fusor, marginalizar(SOFTMAX_CIMA), medidas(0.0))
    fusor.reiniciar()
    resultado = fusor.inferir(marginalizar(SOFTMAX_CIMA), medidas(0.0))
    assert resultado.direcao is Direcao.CENTRO  # ainda não confirmou a mudança


def test_direcao_instantanea_precede_a_estavel():
    """No primeiro frame a estável ainda é centro, mas a instantânea já lê."""
    fusor = FusorParaconsistente(ConfigFusao(frames_para_mudar=3))
    resultado = fusor.inferir(marginalizar(SOFTMAX_CIMA), medidas(0.0))

    assert resultado.direcao is Direcao.CENTRO
    assert resultado.direcao_instantanea is Direcao.CIMA
    assert not resultado.estavel


def test_histerese_desligada_responde_no_primeiro_frame():
    """``frames_para_mudar=1`` é o modo correto para imagens avulsas."""
    fusor = FusorParaconsistente(ConfigFusao(frames_para_mudar=1))
    resultado = fusor.inferir(marginalizar(SOFTMAX_CIMA), medidas(0.0))

    assert resultado.direcao is Direcao.CIMA
    assert resultado.direcao_instantanea is Direcao.CIMA
    assert resultado.estavel


def test_direcoes_coincidem_em_regime_estavel():
    fusor = FusorParaconsistente()
    resultado = assentar(fusor, marginalizar(SOFTMAX_CIMA), medidas(0.0))
    assert resultado.direcao is resultado.direcao_instantanea
    assert resultado.estavel
