"""Testes da calibração por usuário."""

from __future__ import annotations

import json

import pytest

from eyeteractive.calibracao import (
    ALVOS,
    AmostraCalibracao,
    acuracia_por_eixo,
    calibrar,
    carregar_perfil,
    salvar_perfil,
)
from eyeteractive.cnn import marginalizar
from eyeteractive.fusion import ConfigFusao
from eyeteractive.geometry import MedidasOculares

# Ordem das classes: center, down, left, right, up.
SOFTMAX = {
    "centro": (0.94, 0.02, 0.01, 0.02, 0.01),
    "cima": (0.02, 0.01, 0.02, 0.01, 0.94),
    "baixo": (0.02, 0.94, 0.02, 0.01, 0.01),
    "esquerda": (0.02, 0.01, 0.94, 0.01, 0.02),
    "direita": (0.02, 0.01, 0.02, 0.94, 0.01),
}


def medidas(grau_h: float, grau_v: float = 0.0) -> MedidasOculares:
    return MedidasOculares(
        grau_horizontal=grau_h,
        grau_vertical=grau_v,
        abertura=0.30,
        confianca_vertical=1.0,
        centroide_iris=(0.0, 0.0),
    )


def sessao(escala_usuario: float = 25.0, n: int = 6) -> list[AmostraCalibracao]:
    """Sessão sintética em que o usuário tem a excursão horizontal indicada."""
    amostras = []
    for nome, vertical, horizontal in ALVOS:
        grau_h = {"esquerda": -escala_usuario, "direita": escala_usuario}.get(horizontal, 0.0)
        grau_v = {"cima": 3.4, "baixo": -3.4}.get(vertical, 0.0)
        for _ in range(n):
            amostras.append(
                AmostraCalibracao(
                    marginais=marginalizar(SOFTMAX[nome]),
                    medidas=medidas(grau_h, grau_v),
                    vertical_esperado=vertical,
                    horizontal_esperado=horizontal,
                )
            )
    return amostras


def test_alvos_cobrem_os_tres_estados_de_cada_eixo():
    verticais = {v for _, v, _ in ALVOS}
    horizontais = {h for _, _, h in ALVOS}
    assert verticais == {"cima", "centro", "baixo"}
    assert horizontais == {"esquerda", "centro", "direita"}


def test_acuracia_por_eixo_perfeita_em_sessao_limpa():
    v, h = acuracia_por_eixo(sessao(), ConfigFusao())
    assert v == pytest.approx(1.0)
    assert h == pytest.approx(1.0)


def test_acuracia_ignora_histerese():
    """A histerese mede estabilidade temporal; amostras aqui são independentes."""
    com = acuracia_por_eixo(sessao(), ConfigFusao(frames_para_mudar=1))
    sem = acuracia_por_eixo(sessao(), ConfigFusao(frames_para_mudar=8))
    assert com == sem


def test_calibrar_recupera_usuario_de_excursao_pequena():
    """Excursão de ±8 pp satura mal com a escala padrão de 25."""
    amostras = sessao(escala_usuario=8.0)
    resultado = calibrar(amostras)

    assert resultado.acuracia_horizontal >= resultado.acuracia_horizontal_padrao
    assert resultado.acuracia_horizontal == pytest.approx(1.0)
    assert resultado.config.escala_horizontal < ConfigFusao().escala_horizontal


def test_calibrar_nao_piora_em_relacao_ao_padrao():
    for escala in (8.0, 15.0, 25.0, 40.0):
        resultado = calibrar(sessao(escala_usuario=escala))
        assert resultado.ganho_vertical >= -1e-9
        assert resultado.ganho_horizontal >= -1e-9


def test_calibrar_preserva_campos_nao_otimizados():
    base = ConfigFusao(frames_para_mudar=5, peso_cnn_horizontal=0.9)
    resultado = calibrar(sessao(), base=base)
    assert resultado.config.frames_para_mudar == 5
    assert resultado.config.peso_cnn_horizontal == pytest.approx(0.9)


def test_calibrar_desempata_pelo_limiar_mais_alto():
    """Sob empate, exigir mais evidência é a escolha segura em uso assistivo."""
    amostras = sessao()
    resultado = calibrar(amostras, limiares=[0.10, 0.20])
    assert resultado.config.limiar_decisao_vertical == pytest.approx(0.20)


def test_calibrar_rejeita_sessao_vazia():
    with pytest.raises(ValueError):
        calibrar([])


def test_ida_e_volta_do_perfil(tmp_path):
    resultado = calibrar(sessao(escala_usuario=12.0))
    caminho = tmp_path / "perfil.json"
    salvar_perfil(caminho, resultado, usuario="teste")

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    assert dados["usuario"] == "teste"
    assert dados["n_amostras"] == resultado.n_amostras

    assert carregar_perfil(caminho) == resultado.config


def test_config_ignora_chaves_desconhecidas():
    """Perfis de versões futuras não podem quebrar o carregamento."""
    dados = ConfigFusao().para_dicionario() | {"parametro_de_outra_versao": 1}
    assert ConfigFusao.de_dicionario(dados) == ConfigFusao()


def test_config_usa_padrao_para_campo_ausente():
    dados = {"limiar_decisao_vertical": 0.4}
    config = ConfigFusao.de_dicionario(dados)
    assert config.limiar_decisao_vertical == pytest.approx(0.4)
    assert config.escala_horizontal == ConfigFusao().escala_horizontal


def test_escala_horizontal_da_config_afeta_a_decisao():
    """Regressão: a escala precisa vir da config, não da constante do módulo.

    A CNN é desligada de propósito: com ela ativa, a marginal horizontal
    sozinha já acerta e mascararia uma escala geométrica mal ajustada.
    """
    amostras = sessao(escala_usuario=6.0)
    so_geometria = {"peso_cnn_horizontal": 0.0}

    larga = acuracia_por_eixo(
        amostras, ConfigFusao(escala_horizontal=40.0, **so_geometria)
    )[1]
    justa = acuracia_por_eixo(
        amostras, ConfigFusao(escala_horizontal=6.0, **so_geometria)
    )[1]

    assert justa == pytest.approx(1.0)
    assert justa > larga


def test_cnn_sustenta_o_horizontal_quando_a_geometria_satura_mal():
    """Documenta por que a rede entra como segunda fonte do eixo horizontal."""
    amostras = sessao(escala_usuario=6.0)
    config = ConfigFusao(escala_horizontal=40.0)  # escala propositalmente ruim

    com_cnn = acuracia_por_eixo(amostras, config)[1]
    sem_cnn = acuracia_por_eixo(
        amostras, ConfigFusao(escala_horizontal=40.0, peso_cnn_horizontal=0.0)
    )[1]

    assert com_cnn > sem_cnn
