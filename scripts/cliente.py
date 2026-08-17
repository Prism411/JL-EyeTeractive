"""Cliente de referência: envia quadros ao servidor e imprime a direção.

Serve tanto para testar o servidor quanto de especificação executável do
protocolo para quem for implementar o cliente móvel.

Uso::

    python scripts/cliente.py --fonte 0                 # webcam
    python scripts/cliente.py --fonte data/videos/up.mp4
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eyeteractive.io.protocolo import PORTA_PADRAO, enviar_bloco, receber_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--porta", type=int, default=PORTA_PADRAO)
    parser.add_argument("--fonte", default="0", help="índice de webcam ou caminho de vídeo")
    parser.add_argument("--qualidade", type=int, default=80, help="qualidade JPEG (1-100)")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = sem limite")
    args = parser.parse_args()

    fonte: int | str = int(args.fonte) if args.fonte.isdigit() else args.fonte
    captura = cv2.VideoCapture(fonte)
    if not captura.isOpened():
        print(f"erro: não abriu a fonte {args.fonte}", file=sys.stderr)
        return 1

    try:
        conexao = socket.create_connection((args.host, args.porta), timeout=10)
    except OSError as erro:
        print(f"erro: servidor indisponível em {args.host}:{args.porta} ({erro})", file=sys.stderr)
        captura.release()
        return 1

    enviados = 0
    try:
        with conexao:
            while not args.max_frames or enviados < args.max_frames:
                ok, quadro = captura.read()
                if not ok:
                    break

                ok, buffer = cv2.imencode(
                    ".jpg", quadro, [int(cv2.IMWRITE_JPEG_QUALITY), args.qualidade]
                )
                if not ok:
                    continue

                enviar_bloco(conexao, buffer.tobytes())
                resposta = receber_json(conexao)
                enviados += 1

                if resposta.get("direcao"):
                    marca = "" if resposta.get("estavel", True) else "  (transitório)"
                    print(
                        f"{resposta['direcao']:<16} código={resposta['codigo']} "
                        f"conf={resposta['confianca']:.2f}"
                        f"  V={resposta['vertical']['valor']:<8}"
                        f"  H={resposta['horizontal']['valor']:<8}{marca}"
                    )
    except (ConnectionError, OSError) as erro:
        print(f"conexão encerrada: {erro}", file=sys.stderr)
    except KeyboardInterrupt:
        pass
    finally:
        captura.release()

    print(f"\n{enviados} quadros enviados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
