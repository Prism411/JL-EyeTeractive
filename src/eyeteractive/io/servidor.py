"""Servidor de inferência: recebe quadros e devolve a direção do olhar.

Mantém o desenho *server-side* do projeto — o dispositivo cliente só captura e
exibe, enquanto a GPU fica no servidor — mas com o protocolo de
:mod:`eyeteractive.io.protocolo`, sem ``pickle`` e com uma conexão por vez
tratada de forma isolada, de modo que um cliente que caia não derrube o laço
principal.
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..fusion import ConfigFusao
from ..pipeline import PipelineOlhar, codigo_de_direcao
from .protocolo import PORTA_PADRAO, enviar_json, receber_bloco

__all__ = ["ServidorOlhar"]

_log = logging.getLogger(__name__)


class ServidorOlhar:
    """Servidor TCP de inferência do olhar."""

    def __init__(
        self,
        pesos_cnn: str | Path,
        modelo_landmarker: str | Path,
        endereco: str = "0.0.0.0",
        porta: int = PORTA_PADRAO,
        config: Optional[ConfigFusao] = None,
        pular_quadros: int = 1,
        apenas_cardeais: bool = False,
    ) -> None:
        self.endereco = endereco
        self.porta = porta
        self.pular_quadros = max(1, pular_quadros)
        self.apenas_cardeais = apenas_cardeais
        self.pipeline = PipelineOlhar(
            pesos_cnn=pesos_cnn,
            modelo_landmarker=modelo_landmarker,
            config=config,
            modo_video=True,
        )

    def servir(self) -> None:
        """Laço principal: aceita clientes em série, indefinidamente."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as servidor:
            servidor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            servidor.bind((self.endereco, self.porta))
            servidor.listen(1)
            _log.info("escutando em %s:%d", self.endereco, self.porta)

            while True:
                conexao, origem = servidor.accept()
                _log.info("cliente conectado: %s", origem)
                self.pipeline.reiniciar()
                try:
                    self._atender(conexao)
                except ConnectionError as erro:
                    _log.info("cliente %s desconectou: %s", origem, erro)
                except Exception:  # pragma: no cover - resiliência do laço
                    _log.exception("erro ao atender %s", origem)
                finally:
                    conexao.close()
                    _log.info("conexão com %s encerrada", origem)

    def _atender(self, conexao: socket.socket) -> None:
        contador = 0
        inicio = time.monotonic()

        while True:
            bruto = receber_bloco(conexao)
            if not bruto:
                return

            contador += 1
            if contador % self.pular_quadros != 0:
                enviar_json(conexao, {"direcao": None, "codigo": 0, "motivo": "quadro pulado"})
                continue

            quadro = cv2.imdecode(np.frombuffer(bruto, dtype=np.uint8), cv2.IMREAD_COLOR)
            if quadro is None:
                enviar_json(
                    conexao, {"direcao": None, "codigo": 0, "motivo": "quadro inválido"}
                )
                continue

            timestamp_ms = int((time.monotonic() - inicio) * 1000)
            analise = self.pipeline.processar(quadro, timestamp_ms=timestamp_ms)

            if not analise.rosto_detectado:
                enviar_json(
                    conexao, {"direcao": None, "codigo": 0, "motivo": "sem rosto"}
                )
                continue

            resultado = analise.resultado
            enviar_json(
                conexao,
                {
                    "direcao": resultado.direcao.value,
                    "codigo": codigo_de_direcao(resultado.direcao, self.apenas_cardeais),
                    "confianca": round(resultado.confianca, 4),
                    "estavel": resultado.estavel,
                    "vertical": {
                        "valor": resultado.vertical.valor,
                        "certeza_real": round(resultado.vertical.certeza_real, 4),
                        "contradicao": round(resultado.vertical.anotacao.contradicao, 4),
                        "estado": resultado.vertical.estado.value,
                    },
                    "horizontal": {
                        "valor": resultado.horizontal.valor,
                        "certeza_real": round(resultado.horizontal.certeza_real, 4),
                        "contradicao": round(resultado.horizontal.anotacao.contradicao, 4),
                        "estado": resultado.horizontal.estado.value,
                    },
                },
            )

    def fechar(self) -> None:
        self.pipeline.fechar()
