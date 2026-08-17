"""Testes da geometria ocular, incluindo a regressão do sinal vertical."""

from __future__ import annotations

import pytest

from eyeteractive.geometry import (
    ESCALA_VERTICAL,
    abertura_relativa,
    centroide,
    confianca_vertical,
    distancias_ao_ponto,
    graus,
    medir,
)

# Contorno sintético de olho na convenção documentada, centrado em (100, 100):
# 0=canto externo, 1..2=pálpebra superior, 3=canto interno, 4..5=inferior.
CONTORNO = (
    (80.0, 100.0),  # 0 canto externo   (esquerda da imagem)
    (90.0, 92.0),   # 1 superior externo
    (110.0, 92.0),  # 2 superior interno
    (120.0, 100.0), # 3 canto interno   (direita da imagem)
    (110.0, 108.0), # 4 inferior interno
    (90.0, 108.0),  # 5 inferior externo
)


def iris_em(x: float, y: float, raio: float = 4.0):
    """Quatro pontos de íris ao redor de um centro."""
    return ((x - raio, y), (x, y - raio), (x + raio, y), (x, y + raio))


def test_centroide_nao_arredonda():
    assert centroide(((0.0, 0.0), (1.0, 1.0))) == (0.5, 0.5)


def test_distancias_preservam_fracao():
    d = distancias_ao_ponto(((3.0, 4.0),), (0.0, 0.0))
    assert d[0] == pytest.approx(5.0)


def test_graus_exige_seis_distancias():
    with pytest.raises(ValueError):
        graus([1.0, 2.0, 3.0])


def test_olhar_para_cima_produz_grau_vertical_positivo():
    """Regressão do bug central: no protótipo, 'cima' saía negativo.

    A íris subindo se afasta da pálpebra inferior, então o escore de 'cima'
    é a contribuição dos pontos inferiores — e entra com sinal positivo.
    """
    dists = distancias_ao_ponto(CONTORNO, centroide(iris_em(100.0, 94.0)))
    _, grau_v = graus(dists)
    assert grau_v > 0


def test_olhar_para_baixo_produz_grau_vertical_negativo():
    dists = distancias_ao_ponto(CONTORNO, centroide(iris_em(100.0, 106.0)))
    _, grau_v = graus(dists)
    assert grau_v < 0


def test_olhar_centrado_tem_grau_vertical_proximo_de_zero():
    dists = distancias_ao_ponto(CONTORNO, centroide(iris_em(100.0, 100.0)))
    _, grau_v = graus(dists)
    assert abs(grau_v) < 1.0


def test_grau_vertical_e_monotonico_no_deslocamento():
    valores = []
    for y in (106.0, 103.0, 100.0, 97.0, 94.0):
        dists = distancias_ao_ponto(CONTORNO, centroide(iris_em(100.0, y)))
        valores.append(graus(dists)[1])
    assert valores == sorted(valores), "subir a íris deve aumentar o grau vertical"


def test_olhar_para_direita_produz_grau_horizontal_positivo():
    dists = distancias_ao_ponto(CONTORNO, centroide(iris_em(112.0, 100.0)))
    grau_h, _ = graus(dists)
    assert grau_h > 0


def test_olhar_para_esquerda_produz_grau_horizontal_negativo():
    dists = distancias_ao_ponto(CONTORNO, centroide(iris_em(88.0, 100.0)))
    grau_h, _ = graus(dists)
    assert grau_h < 0


def test_eixos_sao_independentes_no_contorno_simetrico():
    """Deslocar só na horizontal não deve mover o grau vertical."""
    dists = distancias_ao_ponto(CONTORNO, centroide(iris_em(112.0, 100.0)))
    _, grau_v = graus(dists)
    assert abs(grau_v) < 0.5


def test_excursao_vertical_e_menor_que_a_horizontal():
    """Documenta em teste a razão de a geometria não resolver o vertical."""
    d_cima = distancias_ao_ponto(CONTORNO, centroide(iris_em(100.0, 94.0)))
    d_direita = distancias_ao_ponto(CONTORNO, centroide(iris_em(112.0, 100.0)))
    assert abs(graus(d_cima)[1]) < abs(graus(d_direita)[0])


def test_abertura_relativa_cai_com_olho_fechado():
    fechado = (
        (80.0, 100.0), (90.0, 99.0), (110.0, 99.0),
        (120.0, 100.0), (110.0, 101.0), (90.0, 101.0),
    )
    assert abertura_relativa(fechado) < abertura_relativa(CONTORNO)


def test_confianca_vertical_zera_com_olho_ocluido():
    assert confianca_vertical(0.10) == 0.0
    assert confianca_vertical(0.35) == 1.0
    assert 0.0 < confianca_vertical(0.23) < 1.0


def test_medir_integra_as_etapas():
    m = medir(CONTORNO, iris_em(100.0, 94.0))
    assert m.grau_vertical > 0
    assert m.abertura > 0
    assert 0.0 <= m.confianca_vertical <= 1.0
    assert m.centroide_iris == pytest.approx((100.0, 94.0))


def test_escala_vertical_cobre_a_excursao_observada():
    """A escala de saturação precisa acomodar o ±3,45 medido em up.mp4."""
    assert ESCALA_VERTICAL >= 3.45
