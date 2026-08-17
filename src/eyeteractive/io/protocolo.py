"""Protocolo de rede entre captura e inferência.

O servidor original desserializava os frames recebidos com ``pickle.loads``.
Como o socket escutava em ``0.0.0.0``, qualquer máquina alcançável na rede
podia enviar um *pickle* forjado e executar código arbitrário no host — o
``pickle`` só embrulhava o buffer JPEG devolvido por ``cv2.imencode``, então
não havia nada a ganhar com ele.

Aqui o quadro trafega como JPEG puro com prefixo de tamanho, e a resposta é
uma linha JSON. Nenhum dado vindo da rede é interpretado como objeto Python.

Formato::

    cliente → servidor : uint32 big-endian (tamanho) + bytes JPEG
    servidor → cliente : uint32 big-endian (tamanho) + JSON UTF-8

O JSON de resposta carrega a direção, o código inteiro do comando e os graus
de certeza dos dois eixos, de modo que o cliente possa aplicar a própria
política de aceitação em vez de receber só um inteiro sem contexto.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

__all__ = [
    "PORTA_PADRAO",
    "TAMANHO_MAXIMO_QUADRO",
    "enviar_bloco",
    "receber_bloco",
    "enviar_json",
    "receber_json",
]

PORTA_PADRAO = 5000

# Teto defensivo: um quadro JPEG legítimo de 1080p comprimido fica bem abaixo
# disso. Sem o limite, um tamanho forjado obrigaria o servidor a alocar
# memória arbitrária antes de qualquer validação.
TAMANHO_MAXIMO_QUADRO = 8 * 1024 * 1024

_CABECALHO = struct.Struct("!I")


def enviar_bloco(conexao: socket.socket, dados: bytes) -> None:
    """Envia um bloco com prefixo de tamanho."""
    if len(dados) > TAMANHO_MAXIMO_QUADRO:
        raise ValueError(f"Bloco de {len(dados)} bytes excede o limite do protocolo.")
    conexao.sendall(_CABECALHO.pack(len(dados)) + dados)


def _receber_exatamente(conexao: socket.socket, quantidade: int) -> bytes:
    """Lê exatamente ``quantidade`` bytes ou levanta ``ConnectionError``."""
    partes = bytearray()
    while len(partes) < quantidade:
        pedaco = conexao.recv(min(65536, quantidade - len(partes)))
        if not pedaco:
            raise ConnectionError("Conexão encerrada pelo par.")
        partes.extend(pedaco)
    return bytes(partes)


def receber_bloco(conexao: socket.socket) -> bytes:
    """Lê um bloco com prefixo de tamanho, validando o limite antes de alocar."""
    (tamanho,) = _CABECALHO.unpack(_receber_exatamente(conexao, _CABECALHO.size))
    if tamanho == 0:
        return b""
    if tamanho > TAMANHO_MAXIMO_QUADRO:
        raise ConnectionError(f"Tamanho anunciado ({tamanho} bytes) excede o limite.")
    return _receber_exatamente(conexao, tamanho)


def enviar_json(conexao: socket.socket, carga: dict[str, Any]) -> None:
    enviar_bloco(conexao, json.dumps(carga, ensure_ascii=False).encode("utf-8"))


def receber_json(conexao: socket.socket) -> dict[str, Any]:
    return json.loads(receber_bloco(conexao).decode("utf-8"))
