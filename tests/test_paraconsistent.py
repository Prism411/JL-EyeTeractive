"""Testes da álgebra paraconsistente Eτ."""

from __future__ import annotations


import pytest

from eyeteractive.paraconsistent import (
    Anotacao,
    EstadoLogico,
    combinar,
    evidencia_de_escore,
    evidencia_de_probabilidades,
    maximizacao,
    minimizacao,
)


def test_graus_basicos():
    a = Anotacao(0.9, 0.1)
    assert a.certeza == pytest.approx(0.8)
    assert a.contradicao == pytest.approx(0.0)


def test_anotacao_satura_fora_do_intervalo():
    a = Anotacao(1.7, -0.4)
    assert a.mu == 1.0
    assert a.lam == 0.0


def test_estado_verdadeiro_e_falso():
    assert Anotacao(0.9, 0.05).estado() is EstadoLogico.VERDADEIRO
    assert Anotacao(0.05, 0.9).estado() is EstadoLogico.FALSO


def test_estado_inconsistente_tem_precedencia_sobre_certeza():
    """Duas fontes afirmando e negando com força não podem virar decisão."""
    a = Anotacao(1.0, 0.8)
    assert a.certeza == pytest.approx(0.2)
    assert a.contradicao == pytest.approx(0.8)
    assert a.estado() is EstadoLogico.INCONSISTENTE


def test_estado_paracompleto_quando_ninguem_se_compromete():
    a = Anotacao(0.1, 0.1)
    assert a.contradicao == pytest.approx(-0.8)
    assert a.estado() is EstadoLogico.PARACOMPLETO


def test_certeza_real_colapsa_sob_contradicao():
    """Gc alto com contradição alta não pode ser reportado como confiança."""
    limpa = Anotacao(0.9, 0.1)
    contraditoria = Anotacao(0.95, 0.95)

    assert limpa.certeza_real > 0.7
    assert contraditoria.certeza == pytest.approx(0.0, abs=1e-9)
    assert contraditoria.certeza_real == pytest.approx(0.0)


def test_certeza_real_preserva_o_sinal():
    assert Anotacao(0.9, 0.05).certeza_real > 0
    assert Anotacao(0.05, 0.9).certeza_real < 0


def test_certeza_real_nunca_excede_certeza():
    for mu in (0.0, 0.25, 0.5, 0.75, 1.0):
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            a = Anotacao(mu, lam)
            assert abs(a.certeza_real) <= abs(a.certeza) + 1e-9


def test_negacao_troca_evidencias():
    a = Anotacao(0.8, 0.2)
    assert a.negada() == Anotacao(0.2, 0.8)
    assert a.negada().certeza == pytest.approx(-a.certeza)


def test_combinar_preserva_o_conflito_entre_fontes():
    """Fontes opostas produzem certeza nula — não uma média confiante."""
    resultado = combinar([Anotacao(0.9, 0.1), Anotacao(0.1, 0.9)])
    assert resultado.certeza == pytest.approx(0.0)
    assert resultado.certeza_real == pytest.approx(0.0)


def test_combinar_respeita_pesos():
    forte = Anotacao(1.0, 0.0)
    fraca = Anotacao(0.0, 1.0)
    resultado = combinar([forte, fraca], [3.0, 1.0])
    assert resultado.mu == pytest.approx(0.75)
    assert resultado.lam == pytest.approx(0.25)


def test_combinar_rejeita_entradas_invalidas():
    with pytest.raises(ValueError):
        combinar([])
    with pytest.raises(ValueError):
        combinar([Anotacao(0.5, 0.5)], [1.0, 2.0])
    with pytest.raises(ValueError):
        combinar([Anotacao(0.5, 0.5)], [0.0])


def test_operadores_de_reticulado():
    a, b = Anotacao(0.8, 0.3), Anotacao(0.4, 0.1)
    assert maximizacao([a, b]) == Anotacao(0.8, 0.1)
    assert minimizacao([a, b]) == Anotacao(0.4, 0.3)


def test_evidencia_de_escore_mapeia_sinal():
    assert evidencia_de_escore(10.0, 10.0).certeza == pytest.approx(1.0)
    assert evidencia_de_escore(-10.0, 10.0).certeza == pytest.approx(-1.0)
    assert evidencia_de_escore(0.0, 10.0).certeza == pytest.approx(0.0)


def test_evidencia_de_escore_satura():
    assert evidencia_de_escore(999.0, 10.0).mu == pytest.approx(1.0)


def test_confianca_baixa_empurra_para_paracompleto():
    """Fonte pouco confiável vira ignorância, não evidência fraca do contrário."""
    confiavel = evidencia_de_escore(10.0, 10.0, confianca=1.0)
    duvidosa = evidencia_de_escore(10.0, 10.0, confianca=0.1)

    assert confiavel.estado() is EstadoLogico.VERDADEIRO
    assert duvidosa.estado() is EstadoLogico.PARACOMPLETO
    assert abs(duvidosa.certeza_real) < abs(confiavel.certeza_real)


def test_evidencia_de_probabilidades_ignora_massa_neutra():
    """A massa de 'centro' fica fora: não apoia nem refuta 'está para cima'."""
    a = evidencia_de_probabilidades(favoravel=0.2, contraria=0.1)
    assert a.mu == pytest.approx(0.2)
    assert a.lam == pytest.approx(0.1)
    assert a.estado() is EstadoLogico.PARACOMPLETO
